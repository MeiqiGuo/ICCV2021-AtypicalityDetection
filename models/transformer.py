"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * integrating relative positional encodings
    * implementing relative-spatial transformer
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import copy
from typing import Optional, Tuple


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, pos: Tensor = None, box: Tensor = None, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            pos: relative positions between pairwise elements of the input sequence (required).
                 The first pairwise dimension is the query; the second dimension is the key.
                 Value is the relative position of key compared with query.
            box: input box bounding box.
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            src: (L, N, Es)
            pos: (N, L, L, Ep)
            box: (L, N, Ep)
            output: (L, N, Es)
            Where L is the sequence length, N is the batch size, Es is the embedding dimension of source,
            Ep is the embedding dimension of positions.
        Note:
            pos is a new input of the function compared to the official pytorch.
        """
        output = src

        for mod in self.layers:
            output = mod(output, pos, box, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        d_pos: the number of expected features in the input relative position (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, d_pos, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu", bias=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadSelfAttentionWithRelativePosition(d_model, d_pos, nhead, dropout=dropout, bias=bias)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src: Tensor, pos: Tensor = None, box: Tensor = None, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            pos: relative positions between pairwise elements of the input sequence (required).
            box: input box bounding box.
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in TransformerEncoderLayer class.
        """
        src2 = self.self_attn(src, pos, box, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiheadSelfAttentionWithRelativePosition(nn.Module):
    r"""Multi-head self-attention mechanism with relation position encodings.
    Args:
        embed_dim: total dimension of the model.
        pos_dim: dimension of input relation position.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
    """

    def __init__(self, embed_dim, pos_dim, num_heads, dropout=0., bias=True):
        super(MultiheadSelfAttentionWithRelativePosition, self).__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = bias

        self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.p_proj_weight = nn.Parameter(torch.Tensor(embed_dim, pos_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        if self.bias:
            self.q_proj_bias = nn.Parameter(torch.empty(embed_dim))
            self.k_proj_bias = nn.Parameter(torch.empty(embed_dim))
            self.v_proj_bias = nn.Parameter(torch.empty(embed_dim))
            self.p_proj_bias = nn.Parameter(torch.empty(embed_dim))

        self._reset_parameters()

        self.output_dropout = nn.Dropout(self.dropout)


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.p_proj_weight)
        nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias:
            nn.init.constant_(self.q_proj_bias, 0.)
            nn.init.constant_(self.k_proj_bias, 0.)
            nn.init.constant_(self.v_proj_bias, 0.)
            nn.init.constant_(self.p_proj_bias, 0.)

    def forward(self, feature: Tensor, pos: Tensor = None, box: Tensor = None, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        feature: input visual features.
        pos: input relative positional features.
        box: input box bounding box.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shape:
        - Inputs:
        - feature: (L, N, Es)
        - pos: (N, L, L, Ep)
        Where L is the sequence length, N is the batch size, Es is the embedding dimension of source,
        Ep is the embedding dimension of positions.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - Outputs:
        - attn_output: :math:`(L, N, Es)`
        - attn_output_weights: :math:`(N, L, L)`.
        """
        seq_len, bsz, fsz = feature.size()
        assert self.embed_dim == fsz
        scaling = float(self.head_dim) ** -0.5

        if self.bias:
            q = F.linear(feature, self.q_proj_weight, self.q_proj_bias)
            k = F.linear(feature, self.k_proj_weight, self.k_proj_bias)
            v = F.linear(feature, self.v_proj_weight, self.v_proj_bias)
            p = F.linear(pos, self.p_proj_weight, self.p_proj_bias)   #(bsz, len, len, embed_dim)
        else:
            q = F.linear(feature, self.q_proj_weight)
            k = F.linear(feature, self.k_proj_weight)
            v = F.linear(feature, self.v_proj_weight)
            p = F.linear(pos, self.p_proj_weight)   #(bsz, len, len, embed_dim)

        # (bsz, len*len, embed_dim), order of the len*len is (1,1),(2,1)...,(len,1),(1,2),...(len,2),...
        p = p.transpose(1,2).contiguous().view(bsz, seq_len*seq_len, self.embed_dim)

        q = q * scaling

        q = q.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)   #(bsz*num_heads, len, head_dim)
        k = k.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        p = p.transpose(0,1).contiguous().view(seq_len * seq_len, bsz * self.num_heads, self.head_dim).transpose(0,1)  #(bsz*num_heads, len*len, head_dim)

        attn_output_weights_a = torch.bmm(q, k.transpose(1, 2)) #(bsz, len, len)
        repeat_q = q.repeat(1, seq_len, 1)  #(bsz*num_heads, len*len, head_dim)
        attn_output_weights_b = torch.einsum('bij,bij->bij', repeat_q, p).sum(dim=2).view(bsz * self.num_heads, seq_len, seq_len).transpose(1,2)

        attn_output_weights = attn_output_weights_a + attn_output_weights_b

        assert list(attn_output_weights.size()) == [bsz * self.num_heads, seq_len, seq_len]
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.output_dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, seq_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if attn_mask is not None:
            #TODO
            assert False

        if key_padding_mask is not None:
            #TODO
            assert False

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, seq_len, seq_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")