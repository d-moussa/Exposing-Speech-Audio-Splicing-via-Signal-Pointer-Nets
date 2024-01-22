import sys

sys.path.insert(0, "")
import torch
import sources.transformer.dataprep as dataprep

import json
from sources.transformer.dataprep import VocabTransform
from sources.transformer.dataprep import create_vocab, sequetial_transforms, tensor_transform, tensor_transform_encoder, \
    SPECIAL_SYMBOLS, collate_fn
from sources.dataset import PickleDataset, PickleDatasetMultiinput
from sources.transformer.transformer_utils import load_checkpoint, save_checkpoint, train_epoch
import sources.transformer.predict as evaluator
from sources.transformer.pointer_transformer import PointerTransformer
from sources.transformer.encoder_transformer import EncoderTransformer
from timeit import default_timer as timer
import os
from os.path import join, isfile
import re
from pathlib import Path

# Models

# SigPointer (proposed)
import configs.model.sig_pointer.config_sig_pointer_1 as pointer_conf_1
import configs.model.sig_pointer.config_sig_pointer_2 as pointer_conf_2
import configs.model.sig_pointer.config_sig_pointer_3 as pointer_conf_3
import configs.model.sig_pointer.config_sig_pointer_4 as pointer_conf_4
import configs.model.sig_pointer.config_sig_pointer_5 as pointer_conf_5

# SigPointer_CM (SigPointer initialized with Transf. parameters from previous work)
import configs.model.sig_pointer_cm.config_sig_pointer_cm_1 as sig_pointer_cm_1
import configs.model.sig_pointer_cm.config_sig_pointer_cm_2 as sig_pointer_cm_2
import configs.model.sig_pointer_cm.config_sig_pointer_cm_3 as sig_pointer_cm_3
import configs.model.sig_pointer_cm.config_sig_pointer_cm_4 as sig_pointer_cm_4
import configs.model.sig_pointer_cm.config_sig_pointer_cm_5 as sig_pointer_cm_5

# Transformer Encoder Classifier Baseline
import configs.model.transf_encoder.config_transf_encoder_1 as transf_encoder_1
import configs.model.transf_encoder.config_transf_encoder_2 as transf_encoder_2
import configs.model.transf_encoder.config_transf_encoder_3 as transf_encoder_3
import configs.model.transf_encoder.config_transf_encoder_4 as transf_encoder_4
import configs.model.transf_encoder.config_transf_encoder_5 as transf_encoder_5

# Data
import configs.data.config_tts_all_05 as tts_config


def create_dirs(m_config):
    Path(m_config.save_model_dir).mkdir(parents=True, exist_ok=True)


def create_pickle_path_dicts(pickle_path, name):
    retVal = {}
    onlyfiles = [f for f in os.listdir(pickle_path) if isfile(join(pickle_path, f))]
    filtered_files = [f for f in onlyfiles if (name in f) and (f.endswith(".pkl"))]
    if not name.endswith("_m"):
        filtered_files = [f for f in filtered_files if ("_m_" not in f)]
    nums = [int(re.findall(r'\d+', f)[-1]) for f in filtered_files]
    for k, v in zip(nums, filtered_files):
        retVal[k] = pickle_path + v
    return retVal


def create_dataloaders(splice_seq_transform, d_config, m_config):
    train_pickle_dict = create_pickle_path_dicts(d_config.dataset_root_path, d_config.dataset_pickle_name_train)
    eval_pickle_dict = create_pickle_path_dicts(d_config.dataset_root_path, d_config.dataset_pickle_name_val)
    test_pickle_dict = create_pickle_path_dicts(d_config.dataset_root_path, d_config.dataset_pickle_name_test)

    if d_config.input_specs_types is None:
        ds_train = PickleDataset(train_pickle_dict, d_config.dataset_path_train, d_config.package_size)
        ds_eval = PickleDataset(eval_pickle_dict, d_config.dataset_path_val, d_config.package_size)
        ds_test = PickleDataset(test_pickle_dict, d_config.dataset_path_test, d_config.package_size)
    else:
        ds_train = PickleDatasetMultiinput(train_pickle_dict, d_config.dataset_path_train, d_config.package_size)
        ds_eval = PickleDatasetMultiinput(eval_pickle_dict, d_config.dataset_path_val, d_config.package_size)
        ds_test = PickleDatasetMultiinput(test_pickle_dict, d_config.dataset_path_test, d_config.package_size)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=m_config.batch_size, shuffle=False, num_workers=0,
                                           collate_fn=collate_fn(splice_seq_transform,
                                                                 encoder=m_config.num_decoder_layers == 0))
    dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=m_config.batch_size, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn(splice_seq_transform,
                                                                encoder=m_config.num_decoder_layers == 0))
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn(splice_seq_transform,
                                                                encoder=m_config.num_decoder_layers == 0))
    return dl_train, dl_eval, dl_test


def create_model(m_config, device):
    # Transf. Encoder Classifier
    if m_config.num_decoder_layers == 0:
        print("INFO -- Constructing Encoder Network")
        transformer = EncoderTransformer(m_config.num_encoder_layers, m_config.emb_size, m_config.nhead,
            m_config.ffn_hid_dim)
    else:  # Transformer Pointer
        print("INFO -- Constructing Pointer Network")
        transformer = PointerTransformer(m_config.num_encoder_layers, m_config.num_decoder_layers, m_config.emb_size,
                                         m_config.nhead, m_config.ffn_hid_dim)

    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform(p)
    transformer = transformer.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataprep.PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=m_config.lr, betas=m_config.betas, eps=m_config.eps)
    return transformer, loss_fn, optimizer


def train_transformer(model, dl_train, dl_val, optimizer, scheduler, loss_fn, m_config, d_config, device, subset=1.0):
    train_losses = []
    eval_losses = []
    loss_increase_count = 0
    best_val_loss = 10000

    for epoch in range(1, m_config.num_epochs + 1):
        start_time = timer()
        train_loss, eval_loss = train_epoch(model, optimizer, dl_train, dl_val, loss_fn, device, subset)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        end_time = timer()
        print("Epoch: ", epoch, " Train Loss: ", train_loss, " Val Mean Dist: ", eval_loss, " Epoc Time: ",
              (end_time - start_time))

        if (eval_loss + m_config.early_stopping_delta) >= best_val_loss:
            loss_increase_count += 1
            print("loss increase count: ", loss_increase_count)
            if loss_increase_count == m_config.early_stopping_wait:
                break
        else:
            loss_increase_count = 0

        checkpoint_save_path = m_config.save_model_dir + m_config.model_name + ".pth"

        if eval_loss < best_val_loss:
            print("Model improved: ", eval_loss, " | ", best_val_loss)
            save_checkpoint(checkpoint_save_path, model, optimizer)
            best_val_loss = eval_loss
    return train_losses, eval_losses


def train(m_config, d_train_config, device="cpu"):
    create_dirs(m_config, d_train_config)

    print(" --- TRAINING: ", d_train_config.config_name, m_config.model_name, " ---")
    print("INFO -- Device: {}".format(device))
    subset = 1

    vocab_transform_dict = {}
    special_symbols = [] if m_config.num_decoder_layers == 0 else SPECIAL_SYMBOLS

    vocab_transform_dict = create_vocab(vocab_transform_dict, special_symbols)
    vocab_transform = VocabTransform(vocab_transform_dict)

    if m_config.num_decoder_layers == 0:  # Encoder model doesnt need s2s flags sos and eos
        splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform_encoder)
    else:
        splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform)
    model, loss_fn, optimizer = create_model(m_config, device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    if m_config.load_checkpoint_path is not None:
        print("LOADING CHECKPOINT")
        model, optimizer, _ = load_checkpoint(m_config.load_checkpoint_path, model, optimizer, device)
        print("Loaded: ", m_config.load_checkpoint_path)

    dl_train, dl_eval, dl_test = create_dataloaders(splice_seq_transform, d_train_config, m_config)
    for dl in [dl_train, dl_eval, dl_test]:
        pkg_size = dl.dataset.package_size
        len_specs = len(dl.dataset.specs[0])
        if not pkg_size == len_specs:
            print("pkg size is false in config, true is: {}".format(len_specs))
        print("package size: {}".format(len_specs))
        print("len_spec_pickle_dict: {}".format(len(dl.dataset.spec_pickle_dict)))
        print("total num. of samples: {}".format(len_specs * len(dl.dataset.spec_pickle_dict)))
        print("spec_pickle_dict: {}".format(dl.dataset.spec_pickle_dict))
        print("-" * 20)
    train_losses, eval_losses = train_transformer(model, dl_train, dl_eval, optimizer, scheduler, loss_fn, m_config,
                                                  d_train_config, device, subset=subset)
    return train_losses, eval_losses


def load_predict(m_config, d_test_config, device="cpu", beam_width=[1]):
    print(" --- PREDICTIONS: ", d_test_config.config_name, m_config.model_name, " ---")

    print("INFO -- Device: {}".format(device))
    vocab_transform_dict = {}
    vocab_transform_dict = create_vocab(vocab_transform_dict, SPECIAL_SYMBOLS, max_size=200)
    vocab_transform = VocabTransform(vocab_transform_dict)
    splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform)

    model, _, _ = create_model(m_config, device)

    loaded_optimizer = torch.optim.Adam(model.parameters(), lr=m_config.lr, betas=m_config.betas, eps=m_config.eps)
    checkpoint_model_name = m_config.model_name + ".pth"
    checkpoint_save_path = os.path.join(m_config.save_model_dir, checkpoint_model_name)
    loaded_model, loaded_optimizer, _ = load_checkpoint(checkpoint_save_path, model, loaded_optimizer, device)

    print("Loaded: ", checkpoint_save_path)
    param_count = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
    print("trainable parameters: ", param_count)

    test_pickle_dict = create_pickle_path_dicts(d_test_config.dataset_root_path, d_test_config.dataset_pickle_name_test)

    if d_test_config.input_specs_types is None:
        ds_test = PickleDataset(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)
    else:
        ds_test = PickleDatasetMultiinput(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn(splice_seq_transform, encoder=False))

    for dl in [dl_test]:
        pkg_size = dl.dataset.package_size
        len_specs = len(dl.dataset.specs[0])
        if not pkg_size == len_specs:
            print("pkg size is false in config, true is: {}".format(len_specs))
        print("package size: {}".format(len_specs))
        print("len_spec_pickle_dict: {}".format(len(dl.dataset.spec_pickle_dict)))
        print("total num. of samples: {}".format(len_specs * len(dl.dataset.spec_pickle_dict)))
        print("spec_pickle_dict: {}".format(dl.dataset.spec_pickle_dict))
        print("-" * 20)
    pred_count = 1 if d_test_config.max_split_points == 1 else 5
    beam_widths = beam_width
    ret_dict = evaluator.evaluate_beam(loaded_model, dl_test, pred_count, beam_widths, device)

    result_name = d_test_config.set_name + "-" + d_test_config.experiment_name + "-" + m_config.model_name + "-" + str(
        d_test_config.min_split_points) + "_" + str(d_test_config.max_split_points) + ".json"
    result_save_path = m_config.save_model_dir + result_name
    with open(result_save_path, "w") as fp:
        json.dump(ret_dict, fp)
    print("test done!")


def load_predict_encoder(m_config, d_test_config, device="cpu"):
    print(" --- PREDICTIONS: ", d_test_config.config_name, m_config.model_name, " ---")

    print("INFO -- Device: {}".format(device))
    vocab_transform_dict = {}
    vocab_transform_dict = create_vocab(vocab_transform_dict, [], max_size=200)  # 45

    vocab_transform = VocabTransform(vocab_transform_dict)
    splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform_encoder)

    model, _, _ = create_model(m_config, device)

    loaded_optimizer = torch.optim.Adam(model.parameters(), lr=m_config.lr, betas=m_config.betas, eps=m_config.eps)

    checkpoint_model_name = m_config.model_name + ".pth"
    checkpoint_save_path = os.path.join(m_config.save_model_dir, checkpoint_model_name)
    loaded_model, loaded_optimizer, _ = load_checkpoint(checkpoint_save_path, model, loaded_optimizer, device)

    print("Loaded: ", checkpoint_save_path)
    param_count = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
    print("trainable parameters: ", param_count)

    test_pickle_dict = create_pickle_path_dicts(d_test_config.dataset_root_path, d_test_config.dataset_pickle_name_test)

    if d_test_config.input_specs_types is None:
        ds_test = PickleDataset(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)
    else:
        ds_test = PickleDatasetMultiinput(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn(splice_seq_transform, encoder=True))

    ret_dict = evaluator.evaluate_encoder(loaded_model, dl_test, device)

    result_name = d_test_config.set_name + "-" + d_test_config.experiment_name + "-" + m_config.model_name + "-" + str(
        d_test_config.min_split_points) + "_" + str(d_test_config.max_split_points) + ".json"
    result_save_path = os.path.join(m_config.save_model_dir, result_name)
    with open(result_save_path, "w") as fp:
        json.dump(ret_dict, fp)
    print("test done!")


if __name__ == '__main__':
    # Train Dataset Configs
    d_train_confs = [
    ]

    # Test Dataset Configs
    d_test_confs = [
        tts_config
    ]

    # Model configs
    m_confs = [pointer_conf_1,
        # pointer_conf_2,
        # pointer_conf_3,
        # pointer_conf_4,
        # pointer_conf_5

        # sig_pointer_cm_1,
        # sig_pointer_cm_2,
        # sig_pointer_cm_3,
        # sig_pointer_cm_4,
        # sig_pointer_cm_5,

        # transf_encoder_1,
        # transf_encoder_2,
        # transf_encoder_3,
        # transf_encoder_4,
        # transf_encoder_5,

    ]

    device = "cuda:0"

    # Uncomment this for training
    # for d_train_conf, m_conf in zip(d_train_confs, m_confs):
    #     train(m_conf, d_train_conf, device)

    beam_width = [1]  # [1, 3, 5, 10, 20] (beam search widths considered in eval script)
    for model in m_confs:
        for data in d_test_confs:
            if model.num_decoder_layers == 0:
                load_predict_encoder(model, data, device=device)
            else:
                load_predict(model, data, device=device, beam_width=beam_width)
