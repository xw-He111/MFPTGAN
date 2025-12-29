import torch
import torch.nn as nn

from unet import Hook, CustomPixelShuffle_ICNR, UnetBlockWide, NormType, custom_conv_layer
from convnext import ConvNeXt
from typing import Optional
from torch import Tensor
from torch.nn import functional as F
import math
from utils import feature_visualization, feature_visualization_single

class TDColor(nn.Module):
    def __init__(self,
                 encoder_name='convnext-l',
                 decoder_name='MultiScaleColorDecoder',
                 num_input_channels=3,
                 input_size=(256, 256),
                 nf=512,
                 last_norm='Weight',   # 这个参数代表什么
                 do_normalize=False,
                 num_queries_l=256,
                 num_queries_ab=256,
                 num_scales=3,
                 dec_layers=9,
                 encoder_from_pretrain=False,
                 visualize_feature_maps=False,
                 ):
        super().__init__()

        self.visualize_feature_maps = visualize_feature_maps

        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'], from_pretrain=encoder_from_pretrain)
        self.encoder.eval()   # 固定预训练参数
        test_input = torch.randn(1, num_input_channels, *input_size) # 检查模型的兼容性，初始化缓冲区
        self.encoder(test_input)  # 检查模型的兼容性，初始化缓冲区

        self.decoder = Decoder(
            self.encoder.hooks,
            nf=nf, 
            last_norm=last_norm,
            num_queries_l=num_queries_l,
            num_queries_ab=num_queries_ab,
            num_scales=num_scales,
            dec_layers=dec_layers,
            decoder_name=decoder_name,
            visualize_feature_maps=self.visualize_feature_maps
        )

        self.refine_net_l = nn.Sequential(custom_conv_layer(num_queries_l + 3, 1, ks=1, use_activ=False, norm_type=NormType.Spectral))  # 融合模块，生成l
        self.refine_net_ab = nn.Sequential(custom_conv_layer(num_queries_ab + 3, 2, ks=1, use_activ=False, norm_type=NormType.Spectral))

        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  # 非模型参数缓冲区  self.mean
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)) # 非模型参数缓冲区  self.std
        
    def normalize(self, img):
        return (img - self.mean) / self.std
    
    def denormalize(self, img):
         return img * self.std + self.mean
    
    def forward(self, x):
        if x.shape[1] == 3:   # NCHW?
            x = self.normalize(x)   # 归一化

        self.encoder(x)
        out_l, out_ab = self.decoder()  # [1, 256, 512, 512], [1, 256, 512, 512]

        if self.visualize_feature_maps:
            feature_visualization_single(out_l, features_num=out_l.shape[1])
            # feature_visualization_single(out_l, features_num=out_l.shape[1])

        coarse_input_l = torch.cat([out_l, x], dim=1)   # [1, 103, 512, 512]
        coarse_input_ab = torch.cat([out_ab, x], dim=1)

        out_l = self.refine_net_l(coarse_input_l)   # [1, 1, 512, 512]
        out_ab = self.refine_net_ab(coarse_input_ab)

        if self.do_normalize:
            out_l = self.denormalize(out_l)
            out_ab = self.denormalize(out_ab)

        return out_l, out_ab
    
class Decoder(nn.Module):
    def __init__(self, 
                 hooks, 
                 nf=512,
                 blur=True,
                 last_norm='Weight',
                 num_queries_l=256,
                 num_queries_ab=256,
                 num_scales=3,
                 dec_layers=9,
                 decoder_name='MultiScaleColorDecoder',
                 visualize_feature_maps=False
                 ):
        super().__init__()

        self.visualize_feature_maps = visualize_feature_maps

        self.hooks = hooks
        self.nf = nf
        self.blur = blur   # 这个参数代表什么，有什么作用？
        self.last_norm = getattr(NormType, last_norm)
        self.decoder_name = decoder_name

        self.layers = self.make_layers()   # PixelModel
        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)

        self.decoder_l = MultiScaleColorDecoder(
            in_channles=[512, 512, 256],
            num_queries=num_queries_l,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )

        self.decoder_ab = MultiScaleColorDecoder(
            in_channles=[512, 512, 256],
            num_queries=num_queries_ab,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )


    def forward(self):
        encoder_feat = self.hooks[-1].feature
        out0 = self.layers[0](encoder_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        out3 = self.last_shuf(out2)

        # if self.visualize_feature_maps:
        #     feature_visualization_single(out0, features_num=out0.shape[1])

            # feature_visualization_single(out1, features_num=out1.shape[1])

            # feature_visualization_single(out2, features_num=out2.shape[1])

            # feature_visualization_single(out3, features_num=out3.shape[1])

        out_l = self.decoder_l([out0, out1, out2], out3)
        out_ab = self.decoder_ab([out0, out1, out2], out3)

        return out_l, out_ab
        
    def make_layers(self):  # PixelModel
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[1]   # NCHW

        in_c = e_in_c

        out_c = self.nf

        setup_hooks = self.hooks[-2::-1]  # 从-2到0，逆序

        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:  # [512， 512， 256]
                out_c = out_c // 2
            decoder_layers.append(UnetBlockWide(
                in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))  # 前一层的输出in_c，对应编码器的输出feature_c，生成一个维度为out_c的输出
            in_c = out_c

        return nn.Sequential(*decoder_layers)
    

class Encoder(nn.Module):
    def __init__(self, encoder_name, hook_names, from_pretrain, **kwarges):
        super().__init__()

        if encoder_name == 'convnext-t' or encoder_name == 'convnext':
            self.arch = ConvNeXt()
        elif encoder_name == 'convnext-s':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == 'convnext-b':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == 'convnext-l':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError
        
        self.encoder_name = encoder_name
        self.hook_names = hook_names
        self.hooks = self.setup_hooks()   

        if from_pretrain:
            self.load_pretrain_model()

    def setup_hooks(self):
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks
    
    def forward(self, x):
        return self.arch(x)
    
    def load_pretrain_model(self):
        if self.encoder_name == 'convnext-t' or self.encoder_name == 'convnext':
            self.load('pretrain/convnext_tiny_22k_224.pth')
            print('Loaded pretrained convnext-t model.')
        elif self.encoder_name == 'convnext-s':
            self.load('pretrain/convnext_small_22k_224.pth')
            print('Loaded pretrained convnext-s model.')
        elif self.encoder_name == 'convnext-b':
            self.load('pretrain/convnext_base_22k_224.pth')
            print('Loaded pretrained convnext-b model.')
        elif self.encoder_name == 'convnext-l':
            self.load('pretrain/convnext_large_22k_224.pth')
            print('Loaded pretrained convnext-l model.')
        else:
            raise NotImplementedError

    def load(self, path):
        if not path:
            raise FileNotFoundError(f"The {path} can not be found.")
        
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        incompatible = self.arch.load_state_dict(checkpoint_state_dict, strict=False)  # 将checkpoint_state_dict中保存的权重加载到模型self.arch中，strict=False允许部分加载

        if incompatible.missing_keys:
            msg = "Some model parameters or buffers are not found in the checkpoint:\n"
            print(str(incompatible.missing_keys))

        if incompatible.unexpected_keys:
            msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
            print(str(incompatible.unexpected_keys))


class MultiScaleColorDecoder(nn.Module):  # 用于处理多尺度特征图并生成颜色相关的输出
    def __init__(self, 
                 in_channles,  # 输入通道列表
                 hidden_dim=256,  # 隐藏层维度
                 num_queries=100,  # 查询向量的数量
                 nheads=8, # 多头注意力头数
                 dim_feedforward=2048,   # FFN中间层维度
                 num_scales=3,  # 特征尺度数量
                 dec_layers=9,  # 解码器层数
                 pre_norm=False, # 是否使用pre-normalization
                 embed_dim=256,  # 输出嵌入维度
                 enforce_input_project=True,  # 是否强制使用输入投影
                 ):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)  # 使用正弦位置编码

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):  # 每个transformer block的结构
            self.transformer_self_attention_layers.append(SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm,))
            self.transformer_cross_attention_layers.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm,))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, 
                    dim_feedforward=dim_feedforward, 
                    dropout=0.0, 
                    normalize_before=pre_norm,))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # 可学习参数
        self.num_queries = num_queries  # 使用可学习的查询向量，由num_queries控制
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)  # 查询特征
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 查询位置嵌入

        # level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)  # 尺度级别嵌入，用于区分不同的尺度

        # input projections   输入的预测，这里就有初始化
        self.input_proj = nn.ModuleList()  # 输入投影
        for i in range(self.num_feature_levels):
            if in_channles[i] != hidden_dim or enforce_input_project:  # 只有当输入维度和隐藏维度不同时才进行投影，或强制投影
                self.input_proj.append(nn.Conv2d(in_channles[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)  # weight初始化
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)   # bias初始化
            else:
                self.input_proj.append(nn.Sequential())  

        # output FFNs
        self.out_embed = MLP(hidden_dim, hidden_dim, embed_dim, 3)  # 使用MLP处理最终输出

    def forward(self, x, img_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):  # num_scale 对每个尺度的输入特征计算位置编码
            pos.append(self.pe_layer(x[i], None).flatten(2))  # 位置编码  N_steps = hidden_dim // 2   self.pe_layer = PositionEmbeddingSine(N_steps, normalzie=True)
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  # 应用输入投影

            # flatten NxCxHxW to HWxNxC  调整张量维度
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # Transformer处理
        # QxNxC 初始化查询向量
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # 位置信息
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)  # query_feat作为初始特征

        for i in range(self.num_layers):  
            level_index = i % self.num_feature_levels  # 逐层处理，交替使用不同的特征，轮询方式
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](output, src[level_index], memory_mask=None, memory_key_padding_mask=None, pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)

            # FFN
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output)  # 应用归一化层
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]  # 调整输出维度
        out_embed = self.out_embed(decoder_output)  # 生成最终嵌入
        out = torch.einsum("bqc,bchw->bqhw", out_embed, img_features)  # 将输出与图像特征结合

        return out

class PositionEmbeddingSine(nn.Module):    # 说实话，没看懂
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x