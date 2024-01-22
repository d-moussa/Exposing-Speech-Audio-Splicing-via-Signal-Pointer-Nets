import torch
from torch import nn as nn

from sources.transformer.dataprep import create_mask
from sources.transformer.transformer_layers_with_attn_mats import CustomTransformerEncoderLayer, \
    CustomTransformerEncoder, \
    CustomTransformerDecoderLayer, CustomTransformerDecoder
from sources.transformer.transformer_utils import TokenConstantMapping, PositionalEncodingIncludingOddEmbs


class PointerTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, dim_feedforward=512, dropout=0.1):
        super(PointerTransformer, self).__init__()

        # init encoder and decoder
        encoder_layers = CustomTransformerEncoderLayer(d_model=emb_size,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout)
        self.encoder = CustomTransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        decoder_layers = CustomTransformerDecoderLayer(d_model=emb_size,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout)
        self.decoder = CustomTransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        print("INFO -- Initializing  Constant Target Embedding in PointerNet")
        self.tgt_token_embedding = TokenConstantMapping(
            emb_size)  # zeros (no advantage distinguishable via learned embedding)
        self.positional_encoding = PositionalEncodingIncludingOddEmbs(emb_size, dropout=dropout)
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask,
                memory_mask=None):
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(self.tgt_token_embedding(tgt))
        memory, _ = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        outs, weights = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask, memory_mask=memory_mask)
        enc_dec_attn = [x[1] for x in weights]
        gen = None  # legacy
        return gen, enc_dec_attn

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(src)
        memory, enc_attn_weights = self.encoder(src_emb, mask=src_mask)
        return memory, enc_attn_weights

    def decode(self, tgt, memory, tgt_mask, memory_mask=None):
        tgt_emb = self.tgt_token_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        return self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

    def feed_batch(self, src, tgt, optimizer, device, usecase="Train"):
        src = src.to(device)
        tgt = tgt.long()
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
        memory_mask = None
        logits, enc_dec_attn = self.forward(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                                            src_padding_mask, memory_mask=memory_mask)

        tgt_out = tgt[1:, :]

        last_layer_attn = torch.mean(enc_dec_attn[-1][:, :, :, :], dim=1)
        attn_dist = last_layer_attn.permute(1, 0, 2)
        tgt_out_onehot = torch.nn.functional.one_hot(tgt_out, src.shape[0]).float()

        distance = torch.tensor(0, dtype=torch.float64).to(tgt_out_onehot.device)
        for i in range(attn_dist.shape[0]):
            distance += self.cosine_loss(attn_dist[i], tgt_out_onehot[i], target=torch.tensor([1]).to(attn_dist.device))
        loss = distance

        if usecase == "Train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if usecase == "Eval":
            acc = 0
            for i in range(attn_dist.shape[0]):
                acc += torch.sum(
                    torch.abs(torch.argmax(attn_dist[i], dim=1) - torch.argmax(tgt_out_onehot[i], dim=1))) / \
                       attn_dist.shape[1]
            loss = acc / attn_dist.shape[0]

        return loss
