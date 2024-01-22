import pathlib

import torch


def save_checkpoint(checkpoint_dir, model, optimizer, misc=None):
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
    print("INFO -- saved model and optimizer")


def load_checkpoint(checkpoint_dir, model, optimizer, device, misc_keys=None):
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            print("ERROR -- Loading optimizer failed")
            optimizer = None
    misc_values = {}
    if misc_keys is not None:
        for m in misc_keys:
            misc_values[m] = checkpoint[m]
    model.to(device)
    print("INFO -- loaded model and optimizer")
    return model, optimizer, misc_values


def feed_batch(src, tgt, model, loss_fn, optimizer, device, usecase="Train"):
    src = src.to(device)
    tgt = tgt.to(device)

    logits = model(src.permute(1, 0, 2).unsqueeze(1))

    tgt_out_bce_onehot_two = torch.zeros_like(logits)
    tgt_out_bce_onehot_two[:, :, 1] = tgt
    tgt_out_bce_onehot_two[:, :, 0][tgt_out_bce_onehot_two[:, :, 1] == 0] = 1
    bce_loss = loss_fn(logits, tgt_out_bce_onehot_two)
    loss = bce_loss

    if usecase == "Train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if usecase == "Eval":
        mean_dist = torch.tensor(0, dtype=torch.float32).to(tgt.device)
        for i in range(tgt.shape[1]):
            tgt_pos = torch.where(tgt[:, i] == 1)[0]
            out_pos = torch.where(torch.argmax(logits[:, i, :], dim=1) == 1)[0]
            # if sequence contains no splice tensor is empty
            tgt_pos = torch.tensor([0]) if not tgt_pos.numel() else tgt_pos
            tgt_pad, out_pad = torch.nn.utils.rnn.pad_sequence([tgt_pos, out_pos], padding_value=0,
                                                               batch_first=True).float()
            mean_dist += torch.mean(torch.abs(tgt_pad - out_pad))
        loss = mean_dist / tgt.shape[1]

    return loss


def train_epoch(model, optimizer, train_dl, val_dl, loss_fn, device, subset=1.0):
    model.train()
    train_losses = 0
    eval_losses = 0

    for uc, dl in zip(["Train", "Eval"], [train_dl, val_dl]):
        batch_num = 0
        total_batches = len(dl)
        subset_break_point = int(total_batches * subset)
        if uc == "Train":
            for src, tgt in dl:
                loss = feed_batch(src, tgt, model, loss_fn, optimizer, device, usecase=uc)
                train_losses += loss.item()
                print("TRAINING - Batch: " + str(batch_num) + "/" + str(total_batches) + ", Loss: " + str(loss.item()),
                      end='\r')
                batch_num += 1
                if subset_break_point <= batch_num:
                    break
        elif uc == "Eval":
            with torch.no_grad():
                model.eval()
                for src, tgt in dl:
                    loss = feed_batch(src, tgt, model, loss_fn, optimizer=None, device=device, usecase=uc)
                    eval_losses += loss.item()
                    print("EVALUATION - Batch: " + str(batch_num) + "/" + str(total_batches) + ", Loss: " + str(
                        loss.item()), end='\r')
                    batch_num += 1

    return train_losses / len(train_dl), eval_losses / len(val_dl)
