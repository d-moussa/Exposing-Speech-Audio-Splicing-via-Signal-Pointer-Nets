import math
import pathlib

import torch
from torch import nn as nn, Tensor


def save_checkpoint(checkpoint_dir, model, optimizer, misc = None):
    p = pathlib.Path(checkpoint_dir)
    p_dir = p.parents[0]
    pathlib.Path(p_dir).mkdir(parents=True, exist_ok=True)
    if misc is not None and type(misc) is dict:
        save_dict = misc
        save_dict["model_state_dict"] = model.state_dict()
        save_dict["optimizer_state_dict"] = optimizer.state_dict()
    else:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
    torch.save(save_dict, checkpoint_dir)
    print("Model and optimizer saved!")


def load_checkpoint(checkpoint_dir, model, optimizer, device, misc_keys=None):
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except:
        print("Unable to load the optimizer!")
    misc_values = {}
    if misc_keys is not None:
        for m in misc_keys:
            misc_values[m] = checkpoint[m]
    model.to(device)
    print("Model and optimizer loaded!")
    return model, optimizer, misc_values


def train_epoch(model, optimizer, train_dl, val_dl, loss_fn, device, subset=1.0):
    train_losses = 0
    eval_losses = 0

    for uc, dl in zip(["Train", "Eval"], [train_dl, val_dl]):
        batch_num = 0
        total_batches = len(dl)
        subset_break_point = int(total_batches*subset)
        if uc == "Train":
            for src, tgt in dl:
                loss = model.feed_batch(src, tgt, optimizer, device, usecase=uc)
                train_losses += loss.item()
                print("TRAINING - Batch: "+str(batch_num)+"/"+str(total_batches)+", Loss: "+str(loss.item()), end='\r')
                batch_num += 1
                if subset_break_point <= batch_num:
                    break
        elif uc == "Eval":
            with torch.no_grad():
                model.eval()
                for src, tgt in dl:
                    loss = model.feed_batch(src, tgt, optimizer=None, device=device, usecase=uc)
                    eval_losses += loss.item()
                    print("EVALUATION - Batch: "+str(batch_num)+"/"+str(total_batches)+", Loss: "+str(loss.item()), end='\r')
                    batch_num += 1

    return train_losses/len(train_dl), eval_losses/len(val_dl)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        embs = self.embedding(tokens.long())
        embs = embs * math.sqrt(self.emb_size)
        return embs


class TokenConstantMapping(nn.Module):
    def __init__(self, emb_size):
        super(TokenConstantMapping, self).__init__()
        self.emb_size = emb_size

    def forward(self, tokens):  # for sig_pointer, only positional encoding is relevant
        return torch.zeros((*tokens.shape, self.emb_size)).to(tokens.device)


class PositionalEncodingIncludingOddEmbs(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncodingIncludingOddEmbs, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        if emb_size % 2 == 1:
            den = den[:-1]
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        addedEmbedding = token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        return self.dropout(addedEmbedding)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos*den)
        pos_embedding[:, 1::2] = torch.cos(pos*den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        addedEmbedding = token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        return self.dropout(addedEmbedding)
