import sys

sys.path.insert(0, "")
import torch

import json
from sources.transformer.dataprep import VocabTransform
from sources.transformer.dataprep import create_vocab, sequetial_transforms, tensor_transform_cnn, collate_fn_cnn
from sources.dataset import PickleDataset, PickleDatasetMultiinput

from sources.baselines.baselines_utils import load_checkpoint, save_checkpoint, train_epoch
from sources.baselines.jadhav import Jadhav
from sources.baselines.zeng import Zeng
from sources.baselines.chuchra import Chuchra
from timeit import default_timer as timer
import os
import sources.transformer.predict as evaluator
from os.path import join, isfile
import re
from pathlib import Path

# Models
import configs.model.cnn_baselines.chuchra.config_chuchra_1 as chuchra_1
import configs.model.cnn_baselines.chuchra.config_chuchra_2 as chuchra_2
import configs.model.cnn_baselines.chuchra.config_chuchra_3 as chuchra_3
import configs.model.cnn_baselines.chuchra.config_chuchra_4 as chuchra_4
import configs.model.cnn_baselines.chuchra.config_chuchra_5 as chuchra_5

import configs.model.cnn_baselines.jadhav.config_jadhav_1 as jadhav_1
import configs.model.cnn_baselines.jadhav.config_jadhav_2 as jadhav_2
import configs.model.cnn_baselines.jadhav.config_jadhav_3 as jadhav_3
import configs.model.cnn_baselines.jadhav.config_jadhav_4 as jadhav_4
import configs.model.cnn_baselines.jadhav.config_jadhav_5 as jadhav_5

import configs.model.cnn_baselines.zeng.config_zeng_1 as zeng_1
import configs.model.cnn_baselines.zeng.config_zeng_2 as zeng_2
import configs.model.cnn_baselines.zeng.config_zeng_3 as zeng_3
import configs.model.cnn_baselines.zeng.config_zeng_4 as zeng_4
import configs.model.cnn_baselines.zeng.config_zeng_5 as zeng_5

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


def create_dataloaders(splice_seq_transform, d_config, m_config, vocab_size):
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
                                           collate_fn=collate_fn_cnn(splice_seq_transform, vocab_size))  # CNNs behave like encoder model
    dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=m_config.batch_size, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn_cnn(splice_seq_transform, vocab_size))
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn_cnn(splice_seq_transform, vocab_size))
    return dl_train, dl_eval, dl_test


def create_model(vocab_size, m_config, device):
    if "zeng" in m_config.model_name:
        print("INFO -- Constructing Zeng")
        baseline_cnn = Zeng(channels=1, alphabet_size=vocab_size, pretrained=False, device=device)
    elif "chuchra" in m_config.model_name:
        print("INFO -- Constructing Chuchra")
        baseline_cnn = Chuchra(alphabet_size=vocab_size, device=device)
    elif "jadhav" in m_config.model_name:
        print("INFO -- Constructing Jadhav")
        baseline_cnn = Jadhav(alphabet_size=vocab_size, device=device)
    baseline_cnn = baseline_cnn.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(baseline_cnn.parameters(), lr=m_config.lr)
    return baseline_cnn, loss_fn, optimizer


def train_baseline(model, dl_train, dl_val, optimizer, loss_fn, m_config, d_config, device, subset=1.0):
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

        checkpoint_model_name = m_config.model_name + ".pth"
        checkpoint_save_path = m_config.save_model_dir + checkpoint_model_name

        if eval_loss < best_val_loss:
            print("Model improved: ", eval_loss, " | ", best_val_loss)
            save_checkpoint(checkpoint_save_path, model, optimizer)
            best_val_loss = eval_loss

    return train_losses, eval_losses

def train(m_config, d_train_config, device="cpu"):
    create_dirs(m_config)

    print(" --- TRAINING: ", d_train_config.config_name, m_config.model_name, " ---")
    print("INFO -- Device: {}".format(device))
    subset = 1

    vocab_transform_dict = {}
    special_symbols = []  # CNNs dont need sos, eos or pad
    vocab_transform_dict = create_vocab(vocab_transform_dict, special_symbols)
    vocab_transform = VocabTransform(vocab_transform_dict)

    splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform_cnn)
    vocab_size = len(vocab_transform_dict)
    baseline_cnn, loss_fn, optimizer = create_model(vocab_size, m_config, device)

    if m_config.load_checkpoint_path is not None:
        print("LOADING CHECKPOINT")
        baseline_cnn, loaded_optimizer, _ = load_checkpoint(m_config.load_checkpoint_path, baseline_cnn, optimizer, device)

        optimizer = loaded_optimizer if loaded_optimizer is not None else optimizer
        print("Loaded: ", m_config.load_checkpoint_path)

    dl_train, dl_eval, dl_test = create_dataloaders(splice_seq_transform, d_train_config, m_config, vocab_size)
    train_losses, eval_losses = train_baseline(baseline_cnn, dl_train, dl_eval, optimizer, loss_fn, m_config, d_train_config, device, subset=subset)

    return train_losses, eval_losses


def load_predict_cnn(m_config, d_test_config, device="cpu"):
    print(" --- PREDICTIONS: ", d_test_config.config_name, m_config.model_name, " ---")
    print("INFO -- Device: {}".format(device))

    vocab_transform_dict = {}
    special_symbols = []  # CNNs dont need sos, eos or pad
    vocab_transform_dict = create_vocab(vocab_transform_dict, special_symbols)
    vocab_transform = VocabTransform(vocab_transform_dict)
    splice_seq_transform = sequetial_transforms(vocab_transform, tensor_transform_cnn)

    vocab_size = len(vocab_transform_dict)
    baseline_cnn, _, _ = create_model(vocab_size, m_config, device)

    loaded_optimizer = torch.optim.Adam(baseline_cnn.parameters(), lr=m_config.lr)

    checkpoint_model_name = m_config.model_name + ".pth"
    checkpoint_save_path = m_config.save_model_dir + checkpoint_model_name
    loaded_model, loaded_optimizer, _ = load_checkpoint(checkpoint_save_path, baseline_cnn, loaded_optimizer,
                                                                 device)

    print("Loaded: ", checkpoint_save_path)
    param_count = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
    print("Number of trainable params: ", param_count)

    test_pickle_dict = create_pickle_path_dicts(d_test_config.dataset_root_path, d_test_config.dataset_pickle_name_test)

    if d_test_config.input_specs_types is None:
        ds_test = PickleDataset(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)
    else:
        ds_test = PickleDatasetMultiinput(test_pickle_dict, d_test_config.dataset_path_test, d_test_config.package_size)

    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn_cnn(splice_seq_transform, vocab_size))

    ret_dict = evaluator.evaluate_cnn(loaded_model, dl_test, device)

    result_name = d_test_config.set_name + "-" + d_test_config.experiment_name + "-" + m_config.model_name + "-" + str(
        d_test_config.min_split_points) + "_" + str(d_test_config.max_split_points) + ".json"
    result_save_path = m_config.save_model_dir + result_name
    with open(result_save_path, "w") as fp:
        json.dump(ret_dict, fp)
    print("test done!")


if __name__ == '__main__':
    # Training data configs
    d_train_confs = [

    ]

    # Test data configs
    d_test_confs = [
        tts_config
    ]

    # model configs
    m_confs = [
        chuchra_1,
        chuchra_2,
        chuchra_3,
        chuchra_4,
        chuchra_5,

        jadhav_1,
        jadhav_2,
        jadhav_3,
        jadhav_4,
        jadhav_5,

        zeng_1,
        zeng_2,
        zeng_3,
        zeng_4,
        zeng_5
    ]

    device = "cuda:0"

    #
    # for d_train_conf, m_conf in zip(d_train_confs, m_confs):
    #     train(m_conf, d_train_conf, device)

    # Uncomment this for prediction generation
    for model in m_confs:
        for data in d_test_confs:
            load_predict_cnn(model, data, device=device)
