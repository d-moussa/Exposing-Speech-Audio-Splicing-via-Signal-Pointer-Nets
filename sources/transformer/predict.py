import numpy as np
import torch
import torch.utils.data as tud

from sources.transformer.dataprep import BOS_IDX, generate_square_subsequent_mask
from sources.transformer.pointer_transformer import PointerTransformer


def beam_search_pointer(model, X, src_mask, predictions=20, beam_width=5):
    with torch.no_grad():
        samples_bs = X.size(1)
        device = next(model.parameters()).device
        enc_attn_weights = None
        attention_scores = []
        if isinstance(model, PointerTransformer):
            memory, enc_attn_weights = model.encode(X, src_mask)
        else:
            memory = model.encode(X, src_mask)
        Y = torch.ones(1, samples_bs).fill_(BOS_IDX).type(torch.long).to(device)
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(Y.size(0), device).type(torch.bool))
        if isinstance(model, PointerTransformer):
            msk = torch.zeros((1, memory.shape[0]), device=device).type(torch.bool)
            msk[0, 0] = True
            out, dec_attn_weights = model.decode(Y, memory, tgt_mask)
        else:
            out = model.decode(Y, memory, tgt_mask)

        vocabulary_size = dec_attn_weights[-1][1].shape[3]
        Y = Y.repeat((beam_width, 1))

        last_layer_attn = torch.mean(dec_attn_weights[-1][1][:, :, :, :], dim=1)  # extract attn layer
        attention_scores.append(last_layer_attn)

        attn_dist = (last_layer_attn.squeeze()).log_softmax(-1)  # get probabilities
        attn_probabilities, next_indices = attn_dist.topk(k=beam_width)  # get top k probs
        next_indices = next_indices.reshape(-1, samples_bs)
        Y = torch.cat((Y, next_indices), axis=-1)

        predictions_iterator = range(predictions - 1)
        for i in predictions_iterator:
            temp_x = X.repeat((1, beam_width, 1)).transpose(0, 1)
            dataset = tud.TensorDataset(temp_x, Y)
            loader = tud.DataLoader(dataset, batch_size=256)
            next_attn_probabilities = []
            iterator = iter(loader)
            for x, y in iterator:
                x = x.transpose(0, 1)
                y = y.transpose(0, 1)
                if isinstance(model, PointerTransformer):
                    memory, enc_attn_weights = model.encode(x, src_mask)
                else:
                    memory = model.encode(x, src_mask)

                memory = memory.to(device)
                tgt_mask = (generate_square_subsequent_mask(y.size(0), device).type(torch.bool))
                if isinstance(model, PointerTransformer):
                    out, dec_attn_weights = model.decode(y, memory, tgt_mask)
                else:
                    out = model.decode(y, memory, tgt_mask)

                last_layer_attn = torch.mean(dec_attn_weights[-1][1][:, :, :, :], dim=1)
                attention_scores.append(last_layer_attn)
                attn_probs = last_layer_attn[:, -1].log_softmax(-1)  # last attention
                next_attn_probabilities.append(attn_probs)

            next_attn_probabilities = torch.cat(next_attn_probabilities, axis=0)
            next_attn_probabilities = next_attn_probabilities.reshape(
                (-1, beam_width, next_attn_probabilities.shape[-1]))

            attn_probabilities = torch.add(attn_probabilities.unsqueeze(-1), next_attn_probabilities)
            attn_probabilities = attn_probabilities.flatten(start_dim=1)
            attn_probabilities, idx_attn = attn_probabilities.topk(k=beam_width, axis=-1)

            next_indices = torch.remainder(idx_attn, vocabulary_size).flatten().unsqueeze(-1).long()

            Y = torch.cat((Y, next_indices), axis=1)

        return Y.reshape(-1, beam_width, Y.shape[-1]), attn_probabilities, enc_attn_weights, attention_scores


def evaluate_beam(model, val_dl, predictions_count, beam_widths, device):
    model.eval()

    ret_dict = {}
    for beam_width in beam_widths:
        print("processing beam width ", str(beam_width))
        i = 0
        ret_dict["beam_" + str(beam_width)] = {}
        for src, tgt in val_dl:
            src = src.to(device)
            src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool)
            src_mask = src_mask.to(device)

            predictions, log_probabilities, enc_attn, dec_attn = beam_search_pointer(model, src, src_mask,
                                                                                     predictions=predictions_count,
                                                                                     beam_width=beam_width)

            probs = torch.nn.functional.softmax(log_probabilities)

            mat_res = None
            predictions = predictions.cpu().numpy()
            predictions = predictions.reshape(predictions.shape[1], predictions.shape[2])
            probs = probs.cpu().numpy().reshape(-1)
            tgt = tgt.cpu().numpy()

            topk_preds = {}
            topk_pred_probs = {}
            for w in range(len(probs)):
                topk_preds[w] = [0, int(np.argmax(mat_res))] if mat_res is not None else [0, predictions[w].reshape(
                    -1).tolist()]  # skippen der BOS
                topk_pred_probs[w] = float(probs[w])
            ret_dict["beam_" + str(beam_width)][i] = {
                "predictions": topk_preds,
                "probabilities": topk_pred_probs,
                "real": tgt.reshape(-1).tolist()
            }
            i += 1
            if i % 2000 == 0:
                print(i, "/", len(val_dl))

    return ret_dict


def evaluate_encoder(model, val_dl, device):
    model.eval()

    ret_dict = {}
    i = 0
    ret_dict["beam_" + str(1)] = {}
    for src, tgt in val_dl:
        src = src.to(device)
        src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool)
        src_mask = src_mask.to(device)

        outs = model.forward(src, src_mask, src_padding_mask=None)

        probs = torch.nn.functional.softmax(outs.squeeze(), dim=1).detach().cpu().numpy()
        preds = np.where(np.argmax(probs, axis=1) == 1)

        tgt = tgt.cpu().numpy()

        topk_preds = {}
        topk_pred_probs = {}

        topk_preds[0] = [0, sorted(preds[0].reshape(
            -1).tolist(), reverse=True)]  # skippen der BOS
        topk_pred_probs[0] = probs[preds[0]].tolist()
        ret_dict["beam_" + str(1)][i] = {
            "predictions": topk_preds,
            "probabilities": topk_pred_probs,
            "real": np.where(tgt == 1)[0].tolist()
        }
        i += 1
        if i % 2000 == 0:
            print(i, "/", len(val_dl))

    return ret_dict


def evaluate_cnn(model, val_dl, device):
    model.eval()

    ret_dict = {}
    i = 0
    ret_dict["beam_" + str(1)] = {}
    for src, tgt in val_dl:
        src = src.to(device)

        outs = model(src.permute(1, 0, 2).unsqueeze(1))

        probs = torch.nn.functional.softmax(outs.squeeze(), dim=1).detach().cpu().numpy()
        preds = np.where(np.argmax(probs, axis=1) == 1)
        tgt = tgt.cpu().numpy()

        topk_preds = {}
        topk_pred_probs = {}

        topk_preds[0] = [0, sorted(preds[0].reshape(
            -1).tolist(), reverse=True)]
        topk_pred_probs[0] = probs[preds[0]].tolist()
        ret_dict["beam_" + str(1)][i] = {
            "predictions": topk_preds,
            "probabilities": topk_pred_probs,
            "real": np.where(tgt == 1)[0].tolist()
        }
        i += 1
        if i % 2000 == 0:
            print(i, "/", len(val_dl))

    return ret_dict
