# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NRTRDecoder(nn.Module):
    """Transformer Decoder block with self attention mechanism.
    Args:
        n_layers (int): Number of attention layers.
        d_embedding (int): Language embedding dimension.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        d_inner (int): Hidden dimension of feedforward layers.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        dropout (float): Dropout rate.
        num_classes (int): Number of output classes :math:`C`.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        padding_idx (int): The index of `<PAD>`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=108,
                 max_seq_len=25,
                 start_idx=106,
                 padding_idx=0,
                 ):
        super().__init__()

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            TFDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        pred_num_class = num_classes - 1  # ignore padding_idx
        self.classifier = nn.Linear(d_model, pred_num_class)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)

    @staticmethod
    def get_pad_mask(seq, pad_idx):

        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def get_subsequent_mask(seq):
        """For masking out the subsequent info."""
        len_s = seq.size(1)
        subsequent_mask = 1 - torch.triu(
            torch.ones((len_s, len_s), device=seq.device), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).bool()

        return subsequent_mask

    def _attention(self, trg_seq, src, src_mask=None):
        init_idx = torch.ones((trg_seq.shape[0]),device=trg_seq.device,dtype=torch.long) * self.start_idx
        trg_seq = torch.cat((init_idx.unsqueeze(1), trg_seq),1)
        # word_target = torch.full_like(trg_seq, 0, device=trg_seq.device, dtype=torch.long)
        # for i in range(trg_seq.shape[0]):
        #     word_target[i][:len(torch.where(trg_seq[i]!=0)[0])] = trg_seq[i][trg_seq[i]!=0]
        #     end_point = min(max(1, len(torch.where(trg_seq[i]!=0)[0])), trg_seq.size(1)-1)
        #     word_target[i][end_point] = 107
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = self.get_pad_mask(
            trg_seq,
            pad_idx=self.padding_idx) & self.get_subsequent_mask(trg_seq)
        output = tgt
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        output = self.layer_norm(output)

        return output

    def _get_mask(self, logit, img_metas):
        valid_ratios = None
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]
        N, T, _ = logit.size()
        mask = None
        if valid_ratios is not None:
            mask = logit.new_zeros((N, T))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    def forward(self, out_enc, targets):
        out_enc = out_enc.flatten(2,3).permute(0,2,1)
        if self.training:
            targets = targets[:,:25]
            return self.forward_train(out_enc, targets)
        else:
            return self.forward_test(out_enc)

    def forward_train(self, out_enc, targets):
        r"""
        Args:
            feat (None): Unused.
            out_enc (Tensor): Encoder output of shape :math:`(N, T, D_m)`
                where :math:`D_m` is ``d_model``.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)`.
        """
        attn_output = self._attention(targets, out_enc)
        outputs = self.classifier(attn_output)
        outputs = outputs[:, :-1, :].contiguous()
        N, L, D = outputs.size()
        loss = self.loss_ce(outputs.flatten(0,1), targets.flatten(0,1))
        # mask = targets != self.padding_idx
        # acc_rec = acc(torch.argmax(outputs, dim=2).view(N, L), targets.view(N, L),
        #               mask, reduce=True)
        # import pdb;pdb.set_trace()
        return loss

    def forward_test(self, out_enc):
        # src_mask = self._get_mask(out_enc, img_metas)
        N = out_enc.size(0)
        init_target_seq = torch.full((N, self.max_seq_len + 1),
                                     self.padding_idx,
                                     device=out_enc.device,
                                     dtype=torch.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):
            decoder_output = self._attention(
                init_target_seq, out_enc)
            # bsz * seq_len * C
            step_result = F.softmax(
                self.classifier(decoder_output[:, step, :]), dim=-1)
            # bsz * num_classes
            outputs.append(step_result)
            _, step_max_index = torch.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index
        outputs = torch.stack(outputs, dim=1)
        init_target_seq = init_target_seq[:,1:]
        return init_target_seq


class TFEncoderLayer(nn.Module):
    """Transformer Encoder Layer.
    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm')
            or ('norm', 'self_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='mmcv.GELU'),
                 operation_order=None):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)

        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = ('norm', 'self_attn', 'norm', 'ffn')

        assert self.operation_order in [('norm', 'self_attn', 'norm', 'ffn'),
                                        ('self_attn', 'norm', 'ffn', 'norm')]

    def forward(self, x, mask=None):
        if self.operation_order == ('self_attn', 'norm', 'ffn', 'norm'):
            residual = x
            x = residual + self.attn(x, x, x, mask)
            x = self.norm1(x)

            residual = x
            x = residual + self.mlp(x)
            x = self.norm2(x)
        elif self.operation_order == ('norm', 'self_attn', 'norm', 'ffn'):
            residual = x
            x = self.norm1(x)
            x = residual + self.attn(x, x, x, mask)

            residual = x
            x = self.norm2(x)
            x = residual + self.mlp(x)

        return x


class TFDecoderLayer(nn.Module):
    """Transformer Decoder Layer.
    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'enc_dec_attn',
            'norm', 'ffn', 'norm') or ('norm', 'self_attn', 'norm',
            'enc_dec_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='mmcv.GELU'),
                 operation_order=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)

        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = ('norm', 'self_attn', 'norm',
                                    'enc_dec_attn', 'norm', 'ffn')
        assert self.operation_order in [
            ('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'),
            ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')
        ]

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        if self.operation_order == ('self_attn', 'norm', 'enc_dec_attn',
                                    'norm', 'ffn', 'norm'):
            dec_attn_out = self.self_attn(dec_input, dec_input, dec_input,
                                          self_attn_mask)
            dec_attn_out += dec_input
            dec_attn_out = self.norm1(dec_attn_out)

            enc_dec_attn_out = self.enc_attn(dec_attn_out, enc_output,
                                             enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm2(enc_dec_attn_out)

            mlp_out = self.mlp(enc_dec_attn_out)
            mlp_out += enc_dec_attn_out
            mlp_out = self.norm3(mlp_out)
        elif self.operation_order == ('norm', 'self_attn', 'norm',
                                      'enc_dec_attn', 'norm', 'ffn'):
            dec_input_norm = self.norm1(dec_input)
            dec_attn_out = self.self_attn(dec_input_norm, dec_input_norm,
                                          dec_input_norm, self_attn_mask)
            dec_attn_out += dec_input

            enc_dec_attn_in = self.norm2(dec_attn_out)
            enc_dec_attn_out = self.enc_attn(enc_dec_attn_in, enc_output,
                                             enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out

            mlp_out = self.mlp(self.norm3(enc_dec_attn_out))
            mlp_out += enc_dec_attn_out

        return mlp_out

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.
    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout layer on attn_output_weights.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.
    Args:
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
    """

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)

        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out, _ = self.attention(q, k, v, mask=mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, len_q, self.dim_v)

        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward module.
    Args:
        d_in (int): The dimension of the input for feedforward
            network model.
        d_hid (int): The dimension of the feedforward
            network model.
        dropout (float): Dropout layer on feedforward output.
        act_cfg (dict): Activation cfg for feedforward module.
    """

    def __init__(self, d_in, d_hid, dropout=0.1, act_cfg=dict(type='Relu')):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):
    """Fixed positional encoding with sine and cosine functions."""

    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Not a parameter
        # Position table of shape (1, n_position, d_hid)
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        self.device = x.device
        x = x + self.position_table[:, :x.size(1)].clone().detach()
        return self.dropout(x)

EPS = 1e-6

def acc_single(a, b, mask):
    ind = mask == 1
    if torch.sum(ind) == 0:
        return 0
    correct = (a[ind] == b[ind]).float()
    acc = torch.sum(correct) / correct.size(0)
    return acc

def acc(a, b, mask, reduce=True):
    batch_size = a.size(0)

    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    acc = a.new_zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        acc[i] = acc_single(a[i], b[i], mask[i])

    if reduce:
        acc = torch.mean(acc)
    return acc