import json
import os
from os.path import join, isfile
from pathlib import Path
import numpy as np
import pandas as pd

END_TOKEN = 2

multi_result_cols = [
    "TestSet", "TestExperiment", "ModelName", "Splits",
    "Jaccard_beam_1", "Jaccard_beam_3", "Jaccard_beam_5", "Jaccard_beam_10", "Jaccard_beam_20",
    "Recall_beam_1", "Recall_beam_3", "Recall_beam_5", "Recall_beam_10", "Recall_beam_20"]

single_result_cols = [
    "TestSet", "TestExperiment", "ModelName", "Splits",
    "Top_1_Acc",  # "Top_2_Acc", "Top_3_Acc", "Top_4_Acc", "Top_5_Acc",
    "Top_1_Delta"]  # , "Top_2_Delta", "Top_3_Delta", "Top_4_Delta", "Top_5_Delta"]

save_path = "outputs/"


def create_pickle_path_dicts(results_dict_path, splice_type):
    onlyfiles = [f for f in os.listdir(results_dict_path) if isfile(join(results_dict_path, f))]
    filtered_files = [f for f in onlyfiles if (splice_type in f) and (f.endswith(".json"))]

    retVal = {}
    for f in filtered_files:
        with open(os.path.join(results_dict_path, f), "r") as fp:
            retVal[f] = json.load(fp)
    return retVal


def check_position_accuracy_delta(result_dict, top_k, s2s=True):
    r_dict = result_dict["beam_1"]
    keys = list(r_dict.keys())

    retVal_delta = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    retVal_acc = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    total_position_frequency = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }
    if not s2s:  # classification task
        pos_fix = 0
    else:
        pos_fix = 1

    for k in keys:  # Samples
        for pos in range(pos_fix, len(r_dict[k]["real"]) - pos_fix):

            real_value = r_dict[k]["real"][pos]
            total_position_frequency[pos + (1 - pos_fix)] += 1
            for i in range(1, top_k + 1):  # Top K
                preds_col = []
                for ii in range(i):
                    if not s2s:
                        pred_value = r_dict[k]["predictions"][str(ii)][1]
                        if pred_value == []:
                            pred_value = [0]
                    else:
                        pred_value = r_dict[k]["predictions"][str(ii)][pos]
                    preds_col.append(pred_value)

                preds_col = np.asarray(preds_col)
                if real_value in preds_col:
                    if i in retVal_acc[pos + (1 - pos_fix)].keys():
                        retVal_acc[pos + (1 - pos_fix)][i] += 1
                    else:
                        retVal_acc[pos + (1 - pos_fix)][i] = 1

                preds_col = np.abs(preds_col - real_value)
                min_delta = np.min(preds_col)

                if i in retVal_delta[pos + (1 - pos_fix)].keys():
                    retVal_delta[pos + (1 - pos_fix)][i].append(min_delta)
                else:
                    retVal_delta[pos + (1 - pos_fix)][i] = [min_delta]

    for splice in retVal_delta.keys():
        for k in retVal_delta[splice].keys():
            retVal_delta[splice][k] = np.mean(retVal_delta[splice][k])

    for k in retVal_acc.keys():
        freq = total_position_frequency[k]
        for t_k in retVal_acc[k].keys():
            retVal_acc[k][t_k] /= freq

    return retVal_acc, retVal_delta


def calc_jaccard(result_dict, s2s=True, delta=1):
    sample_counter = 0
    beam_keys = list(result_dict.keys())
    retVal = {}

    for beam in beam_keys:
        hit_list = []
        treffer_total = 0

        keys = list(result_dict[beam].keys())
        for k in keys:

            if s2s:
                pred_set = np.asarray(result_dict[beam][k]["predictions"]["0"][1])
                pred_set = pred_set[:(pred_set == END_TOKEN).nonzero()[0][0]] if END_TOKEN in pred_set else pred_set
            else:
                pred_set = result_dict[beam][k]["predictions"]["0"][1]
                if pred_set == []:
                    pred_set = np.array([0.])
                else:
                    pred_set = np.array(pred_set)

            pred_set = np.unique(pred_set)
            r_set = np.asarray(result_dict[beam][k]["real"])

            if s2s:  # sos (0) and eos (END_TOKEN)
                r_set = r_set[(r_set != 0) & (r_set != END_TOKEN)]
                pred_set = pred_set[(pred_set != 0) & (pred_set != END_TOKEN)]
            else:
                if r_set.shape[0] == 0:
                    r_set = np.array([0])
            sample_counter += 1

            r_set = np.ceil(r_set / delta)
            pred_set = np.ceil(pred_set / delta)

            total_set = np.concatenate([r_set, pred_set])
            total_set = np.unique(total_set)
            pred_set = np.unique(pred_set)
            r_set = np.unique(r_set)

            a_cat_b = np.concatenate([r_set, pred_set])
            a_cat_b_unique, counts = np.unique(a_cat_b, return_counts=True)
            intersection = a_cat_b_unique[np.where(counts > 1)]

            hits = len(intersection)
            treffer_total += hits
            hits = hits / len(total_set)
            hit_list.append(hits)
        retVal[beam] = np.mean(hit_list)
    print("INFO -- Number of evaluated samples: {}".format(sample_counter))
    return retVal


def calc_recall(result_dict, s2s=True, delta=1):
    sample_counter = 0
    beam_keys = list(result_dict.keys())
    retVal = {}

    for beam in beam_keys:
        hit_list = []
        treffer_total = 0

        keys = list(result_dict[beam].keys())
        for k in keys:

            if s2s:
                pred_set = np.asarray(result_dict[beam][k]["predictions"]["0"][1])
                pred_set = pred_set[:(pred_set == END_TOKEN).nonzero()[0][0]] if END_TOKEN in pred_set else pred_set
            else:
                pred_set = result_dict[beam][k]["predictions"]["0"][1]
                if pred_set == []:
                    pred_set = np.array([0.])
                else:
                    pred_set = np.array(pred_set)

            pred_set = np.unique(pred_set)
            r_set = np.asarray(result_dict[beam][k]["real"]) if result_dict[beam][k]["real"] != [] else np.array([0.])

            if s2s:  # encoder doesnt have sos (0) and eos (END_TOKEN)
                r_set = r_set[(r_set != 0) & (r_set != END_TOKEN)]
                pred_set = pred_set[(pred_set != 0) & (pred_set != END_TOKEN)]
            else:
                if r_set.shape[0] == 0:
                    r_set = np.array([0])

            sample_counter += 1

            r_set = np.ceil(r_set / delta)
            pred_set = np.ceil(pred_set / delta)

            pred_set = np.unique(pred_set)
            r_set = np.unique(r_set)

            a_cat_b = np.concatenate([r_set, pred_set])
            a_cat_b_unique, counts = np.unique(a_cat_b, return_counts=True)
            intersection = a_cat_b_unique[np.where(counts > 1)]

            hits = len(intersection)
            treffer_total += hits
            hits = hits / len(r_set)
            hit_list.append(hits)
        retVal[beam] = np.mean(hit_list)
    print("INFO -- Number of evaluated samples: {}".format(sample_counter))
    return retVal


def split_name_in_parts(name):
    parts = name.split("-")
    test_set = parts[0]
    test_experiment = parts[1]
    model_name = parts[2]
    splits = parts[3][:-5]
    return test_set, test_experiment, model_name, splits


def main(bin, results_path, pointer_task=True, splice_type="0_5"):
    print("INFO -- Evaluating {}-Task".format("Pointer" if pointer_task else "Classification"))
    print("INFO -- Error Range: Binning Targets with Multiples of {}".format(bin))

    result_files = create_pickle_path_dicts(results_path, splice_type)
    keys = list(result_files.keys())

    result_df = []
    for k in keys:
        print(k)
        test_set, test_experiment, model_name, splits = split_name_in_parts(k)
        if splice_type == "0_1":
            acc_dict, delta_dict = check_position_accuracy_delta(result_files[k], top_k=1, s2s=pointer_task)
            if acc_dict[1] == {}:
                acc_dict[1][1] = 0

            result_df.append([
                test_set, test_experiment, model_name, splits,
                acc_dict[1][1],  # acc_dict[1][2], acc_dict[1][3], acc_dict[1][4], acc_dict[1][5],
                delta_dict[1][1]  # , delta_dict[1][2], delta_dict[1][3], delta_dict[1][4], delta_dict[1][5]
            ])
        else:
            try:
                jaccard_dict = calc_jaccard(result_files[k], s2s=pointer_task, delta=bin)
                recall_dict = calc_recall(result_files[k], s2s=pointer_task, delta=bin)
                result_df.append([
                    test_set, test_experiment, model_name, splits,
                    jaccard_dict["beam_1"] if "beam_1" in jaccard_dict else 0,
                    jaccard_dict["beam_3"] if "beam_3" in jaccard_dict else 0,
                    jaccard_dict["beam_5"] if "beam_5" in jaccard_dict else 0,
                    jaccard_dict["beam_10"] if "beam_10" in jaccard_dict else 0,
                    jaccard_dict["beam_20"] if "beam_20" in jaccard_dict else 0,
                    recall_dict["beam_1"] if "beam_1" in recall_dict else 0,
                    recall_dict["beam_3"] if "beam_3" in recall_dict else 0,
                    recall_dict["beam_5"] if "beam_5" in recall_dict else 0,
                    recall_dict["beam_10"] if "beam_10" in recall_dict else 0,
                    recall_dict["beam_20"] if "beam_20" in recall_dict else 0,
                ])

            except:
                print("Failed: ", k)

    if splice_type == "0_1":
        result_df = pd.DataFrame(result_df, columns=single_result_cols)
        result_df.to_csv(
            save_path + splice_type + "_evaluation_model_{}_bin_{}.csv".format(os.path.basename(results_path), bin),
            index=False, sep=";")
    else:
        result_df = pd.DataFrame(result_df, columns=multi_result_cols)
        result_df.to_csv(
            save_path + splice_type + "_evaluation_model_{}_bin_{}.csv".format(os.path.basename(results_path), bin),
            index=False, sep=";")


if __name__ == "__main__":
    pointer_task = True  # set True for pointer models and False for classification models!
    paths = ["models/sig_pointer"]  # list of paths to directories where .json predictions of models are stored
    bins = [1]  # list of frame bins within which a predicted splice is counted as predicted correctly (see paper).
                # bin == 1 means no error tolerance (exact localization)
    splice_type = "0_5"  # min_max occurring splices in dataset (matches prediction file names)
    paths = [Path(p) for p in paths]

    for p in paths:
        for d in bins:
            main(bin=d, results_path=p, pointer_task=pointer_task, splice_type=splice_type)
