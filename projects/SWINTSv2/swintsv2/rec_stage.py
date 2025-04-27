import torch
from torch import nn, Tensor
from .FocalTransformer import FocalTransformerBlock
from .transformer import PositionalEncoding
from .roi_seq_predictors import SequencePredictor
import torch.nn.functional as F
from .litv2 import LITv2

class DynamicConv_v2(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SWINTS.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SWINTS.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SWINTS.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)


        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ELU(inplace=True)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (rec_resolution, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        del parameters
        try:
            features = torch.bmm(features, param1)
        except:
            import pdb;pdb.set_trace()
        del param1
        features = self.norm1(features)
        features = self.activation(features)
        features = torch.bmm(features, param2)

        del param2

        features = self.norm2(features)
        features = self.activation(features)

        return features

class REC_STAGE(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.2, activation="relu"):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv_v2(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ELU(inplace=True)

        self.feat_size = cfg.MODEL.REC_HEAD.POOLER_RESOLUTION
        self.rec_batch_size = cfg.MODEL.REC_HEAD.BATCH_SIZE
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.TLSAM =  nn.Sequential(
            FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),
                FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4)
                 )

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=(self.feat_size[0]//4)*(self.feat_size[1]//4))
        num_channels = d_model
        in_channels = d_model
        mode = 'nearest'
        self.k_encoder = nn.Sequential(
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder_det = nn.Sequential(
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(self.feat_size[0], self.feat_size[1]), mode=mode)
        )
        self.k_decoder_rec = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
        )
        self.seq_encoder = LITv2(pretrain_img_size=224,
                                patch_size=4,
                                in_chans=256,
                                embed_dim=256,
                                depths=[2, 4],
                                num_heads=[4, 8],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.,
                                attn_drop_rate=0.,
                                drop_path_rate=0.2,
                                norm_layer=nn.LayerNorm,
                                ape=False,
                                patch_norm=True,
                                out_indices=(0, 1),
                                frozen_stages=-1,
                                use_checkpoint=False,
                                has_msa=[1, 1],
                                alpha=0.5,
                                local_ws=[4, 2])
        
        self.seq_decoder = SequencePredictor(cfg, d_model)
        self.rescale = nn.Upsample(size=(self.feat_size[0], self.feat_size[1]), mode="bilinear", align_corners=False)

    def forward(self, roi_features, pro_features, gt_masks, N, nr_boxes, idx=None, targets=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """
        features = []
        roi_features = self.seq_encoder(roi_features)
        k = roi_features
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        n,c,h,w = k.size()
        k = k.view(n, c, -1).permute(2, 0, 1)
       # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)

        del pro_features2

        pro_features = self.norm1(pro_features)
   #     # inst_interact.
        if idx:
            pro_features = pro_features.permute(1, 0, 2)[idx][:self.rec_batch_size]
            # pro_features = pro_features.repeat(2,1)[:self.rec_batch_size]
        else:
            pro_features = pro_features.permute(1, 0, 2)
        pro_features = pro_features.reshape(1, -1, self.d_model)
        pro_features2 = self.inst_interact(pro_features, k)
        pro_features = k.permute(1,0,2) + self.dropout2(pro_features2)

        del pro_features2

        obj_features = self.norm2(pro_features)

   #     # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)

        del obj_features2
        obj_features = self.norm3(obj_features)
        obj_features = obj_features.permute(1,0,2)
        obj_features = self.pos_encoder(obj_features)
        obj_features = self.transformer_encoder(obj_features)
        obj_features = obj_features.permute(1,2,0)
        n,c,w = obj_features.shape
        obj_features = obj_features.view(n,c,self.feat_size[0]//4,self.feat_size[1]//4)
        obj_features = obj_features
        k = k.permute(1,2,0)
        k = k.view(n,c,self.feat_size[0]//4,self.feat_size[1]//4)
        k_rec = k*obj_features.sigmoid()
        k_rec = self.k_decoder_rec[0](k_rec)
        k_rec = k_rec + features[0]

        k_det = obj_features
        k_det = self.k_decoder_det[0](k_det)
        k_det = k_det + features[0]
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_rec[1](k_rec) + roi_features
        k_det = self.k_decoder_det[1](k_det) + roi_features
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_det[-1](k_rec)
        k_rec = k_rec.flatten(-2,-1).permute(0,2,1)
        k_rec = self.TLSAM(k_rec)
        k_rec = k_rec.permute(0,2,1).view(n,c,self.feat_size[0],self.feat_size[1])
        gt_masks = self.rescale(gt_masks.unsqueeze(1))
        k_rec = k_rec*gt_masks
        attn_vecs, decoded_scores = self.seq_decoder(k_rec, targets,targets)
        return attn_vecs, decoded_scores 

def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer_worelu(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                   mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, in_c, k, s, p),
                         nn.BatchNorm2d(in_c),
                         nn.ReLU(True),
                         nn.Conv2d(in_c, out_c, k, s, p))

class Attention(nn.Module):
    def __init__(self, in_channels=512, max_length=25, n_feature=256):
        super().__init__()
        self.max_length = max_length

        self.f0_embedding = nn.Embedding(max_length, in_channels)
        self.w0 = nn.Linear(max_length, n_feature)
        self.wv = nn.Linear(in_channels, in_channels)
        self.we = nn.Linear(in_channels, max_length)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.rec_cls = nn.Linear(in_channels, 107)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, enc_output, targets=None):
        bs, c, h, w = enc_output.shape
        enc_output = enc_output.permute(0, 2, 3, 1).flatten(1, 2)
        reading_order = torch.arange(self.max_length, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)  # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)  # b,25,512
        t = self.w0(reading_order_embed.permute(0, 2, 1))  # b,512,256
        t = self.active(t.permute(0, 2, 1) + self.wv(enc_output))  # b,256,512

        attn = self.we(t)  # b,256,25
        attn = self.softmax(attn.permute(0, 2, 1))  # b,25,256
        g_output = torch.bmm(attn, enc_output)  # b,25,512
        res = self.rec_cls(g_output)
        if targets:
            loss = self.loss(res.flatten(0,1), targets[:, :25].flatten(0,1))
            return loss, attn.view(*attn.shape[:2], h, w)
        return res, attn.view(*attn.shape[:2], h, w)

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.downconv = nn.Conv2d(dim+25, dim, 1, 1)

    def forward(self, x):
        x = self.downconv(x)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
