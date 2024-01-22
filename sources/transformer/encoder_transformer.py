import sys

import torch
import torch.nn as nn

from sources.transformer.dataprep import create_mask
from sources.transformer.transformer_layers_with_attn_mats import CustomTransformerEncoderLayer, \
    CustomTransformerEncoder
from sources.transformer.transformer_utils import PositionalEncodingIncludingOddEmbs

sys.path.insert(0, "")
bce_with_logits = torch.nn.BCEWithLogitsLoss()


class EncoderTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 emb_size,
                 nhead,
                 dim_feedforward=512,
                 dropout=0.1):
        super(EncoderTransformer, self).__init__()

        encoder_layers = CustomTransformerEncoderLayer(d_model=emb_size,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout)
        self.encoder = CustomTransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        print("INFO -- Initializing Self-Attention EncoderNet")
        self.decoder = None
        self.enc_mem_projection = nn.Linear(emb_size, 2)
        self.positional_encoding_src = PositionalEncodingIncludingOddEmbs(emb_size, dropout=dropout)

    def forward(self, src, src_mask, src_padding_mask):
        src_emb = self.positional_encoding_src(src)
        memory, weights = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        outs = self.enc_mem_projection(memory)
        return outs

    def feed_batch(self, src, tgt, loss_fn, optimizer, device, usecase="Train"):
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask, _, src_padding_mask, _ = create_mask(src, tgt, device)
        outs = self.forward(src, src_mask, src_padding_mask)

        tgt_out_bce_onehot_two = torch.zeros_like(outs)
        tgt_out_bce_onehot_two[:, :, 1] = tgt
        tgt_out_bce_onehot_two[:, :, 0][tgt_out_bce_onehot_two[:, :, 1] == 0] = 1
        bce_loss = bce_with_logits(outs, tgt_out_bce_onehot_two)
        loss = bce_loss

        if usecase == "Train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if usecase == "Eval":
            mean_dist = torch.tensor(0, dtype=torch.float32).to(tgt.device)
            for i in range(tgt.shape[1]):
                tgt_pos = torch.where(tgt[:, i] == 1)[0]
                out_pos = torch.where(torch.argmax(outs[:, i, :], dim=1) == 1)[0]

                # if sequence contains no splice tensor is empty
                tgt_pos = torch.tensor([0]) if not tgt_pos.numel() else tgt_pos
                tgt_pad, out_pad = torch.nn.utils.rnn.pad_sequence([tgt_pos, out_pos], padding_value=0,
                                                                   batch_first=True).float()
                mean_dist += torch.mean(torch.abs(tgt_pad - out_pad))
            loss = mean_dist / tgt.shape[1]
        return loss
