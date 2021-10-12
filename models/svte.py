"""
Relative-Spatial Transformer Encoder.
"""
import torch
from torch import nn
from .transformer import TransformerEncoder, TransformerEncoderLayer


class MaskVTE(nn.Module):
    """ Transformer Encoder with masked regions as input. """
    def __init__(self, d_model, d_feat, d_pos, nhead, num_layers, dropout=0.1, norm=None):
        super(MaskVTE, self).__init__()

        # Object feature encoding
        self.visn_fc = nn.Linear(d_feat, d_model)
        self.visn_layer_norm = nn.LayerNorm(d_model, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(d_pos, d_model)
        self.box_layer_norm = nn.LayerNorm(d_model, eps=1e-12)

        self.dropout = nn.Dropout(dropout)

        # Mask feature
        self.mask_feature = nn.Parameter(torch.zeros(d_feat))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm)

        self.output_layer = nn.Linear(d_model, d_feat)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.mask_feature)

    def forward(self, scr, pos, mask=None):
        """

        :param scr: input region features, (seq_len, bsz, d_feat)
        :param pos: input boxes, (seq_len, bsz, d_pos)
        :param mask: id of mask for regions, (5, bsz)
        :return: (5, bsz, d_feat)
        """
        if mask is not None:
            # mask randomly selected regions, those region features are predicted targets
            target = self._get_mask_part(scr, mask)
            scr[mask, list(range(mask.shape[1])), :] = self.mask_feature

        x = self.visn_fc(scr)
        x = self.visn_layer_norm(x)
        y = self.box_fc(pos)
        y = self.box_layer_norm(y)
        input_feats = (x + y) / 2
        input_feats = self.dropout(input_feats)

        encoder_output = self.encoder(input_feats)  #(seq_len, bsz, d_model)

        if mask is not None:
            output = self.output_layer(encoder_output)  # (seq_len, bsz, d_feat)
            # only output masked regions
            output = self._get_mask_part(output, mask)
            assert output.shape == target.shape, "Output and target shape is not consistent"
            return output, target
        else:
            return encoder_output

    def _get_mask_part(self, feat, mask):
        return feat[mask, list(range(mask.shape[1])), :]


class MaskSVTE(nn.Module):
    """ Relative-Spacial Transformer Encoder with masked regions as input. """
    def __init__(self, d_model, d_feat, d_pos, nhead, num_layers, dropout=0.1, norm=None):
        super(MaskSVTE, self).__init__()

        # Object feature encoding
        self.visn_fc = nn.Linear(d_feat, d_model)
        self.visn_layer_norm = nn.LayerNorm(d_model, eps=1e-12)

        self.dropout = nn.Dropout(dropout)

        # Mask feature
        self.mask_feature = nn.Parameter(torch.zeros(d_feat))

        encoder_layer = TransformerEncoderLayer(d_model, d_pos, nhead, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, norm)

        self.output_layer = nn.Linear(d_model, d_feat)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.mask_feature)

    def forward(self, scr, pos, mask=None):
        """

        :param scr: input region features, (seq_len, bsz, d_feat)
        :param pos: input boxes, (bsz, seq_len, seq_len, d_pos)
        :param mask: id of mask for regions, (mask_num, bsz)
        :return: (mask_num, bsz, d_feat)
        """
        if mask is not None:
            # mask randomly selected regions, those region features are predicted targets
            target = self._get_mask_part(scr, mask)
            scr[mask, list(range(mask.shape[1])), :] = self.mask_feature

        input_feats = self.visn_fc(scr)
        input_feats = self.visn_layer_norm(input_feats)
        input_feats = self.dropout(input_feats)

        encoder_output = self.encoder(input_feats, pos)  #(seq_len, bsz, d_model)

        if mask is not None:
            output = self.output_layer(encoder_output)  # (seq_len, bsz, d_feat)
            # only output masked regions
            output = self._get_mask_part(output, mask)
            assert output.shape == target.shape, "Output and target shape is not consistent"
            return output, target
        else:
            return encoder_output

    def _get_mask_part(self, feat, mask):
        return feat[mask, list(range(mask.shape[1])), :]

