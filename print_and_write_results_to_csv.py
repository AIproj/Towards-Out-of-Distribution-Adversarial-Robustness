########################
# PRINT RESULTS TO CSV #
########################


import os
import numpy as np
# import torch
import argparse
from distutils.util import strtobool

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("--raw", type=lambda x: bool(strtobool(x)), help="Optional. Does not write to CSV and prints full dict of results. Default: False.")
argument_parser.add_argument("--data", required=True, type=str, choices=["attacks_Lp", "attacks_non_Lp", "SVHN", "CIFAR100", "CIFAR-10-C"], help="Chooses dataset of results to print and save as CSV.")

# Prints saved results from folder ./results based on data used as args.
def print_raw_results(data):
    if data == "attacks_non_Lp":
        RESULTS_PATH = "results/CIFAR10/test_accs_non_Lp/"
    elif data == "attacks_Lp":
        RESULTS_PATH = "results/CIFAR10/test_accs/"
    elif data == "SVHN":
        RESULTS_PATH = "results/SVHN/test_accs/"
    elif data == "CIFAR100":
        RESULTS_PATH = "results/CIFAR100/test_accs/"
    elif data == "CIFAR-10-C":
        RESULTS_PATH = "results/CIFAR-10-C/test_accs/"
    else:
        raise ValueError(data + "is not a supported arg.")
    for root, dirs, files in os.walk(RESULTS_PATH):
        model_filenames = files
        model_paths = [RESULTS_PATH + file for file in files]
    for path in model_paths:
        temp = np.load(path, allow_pickle = True).item()
        results = {}
        for k, v in temp.items():
            if "bool" in k.split('_'):
                continue
            results[k] = v
        # print(results['model_name'])
        print(path, results, '\n\n')

def results_to_csv(data):
    import pandas
    if data == "attacks_non_Lp":
        RESULTS_PATH = "results/CIFAR10/test_accs_non_Lp/"
        ignore_keys = ["num_test_samples", "num_attack_restarts", "model_name", "seen_attacks", "unseen_attacks", "always_unseen_attacks", "skipped_domains_worst_case"]
        model_name_key = "model_name"
    elif data == "attacks_Lp":
        RESULTS_PATH = "results/CIFAR10/test_accs/"
        ignore_keys = ["num_test_samples", "num_attack_restarts", "model_name", "seen_attacks", "unseen_attacks", "always_unseen_attacks", "skipped_domains_worst_case"]
        model_name_key = "model_name"
    elif data == "SVHN":
        RESULTS_PATH = "results/SVHN/test_accs/"
        model_name_key = "base_model"
    elif data == "CIFAR100":
        RESULTS_PATH = "results/CIFAR100/test_accs/"
        model_name_key = "base_model"
    elif data == "CIFAR-10-C":
        RESULTS_PATH = "results/CIFAR-10-C/test_accs/"
        ignore_keys = ["top_k"]
        model_name_key = "model_name"
        accuracies_key = "topk_accuracies"
    else:
        raise ValueError(data + "is not a supported arg.")
    for root, dirs, files in os.walk(RESULTS_PATH):
        model_paths = [RESULTS_PATH + file for file in files]
    results_processed = {}
    for path in model_paths:
        if not path.endswith(".npy"):
            continue
        temp = np.load(path, allow_pickle = True).item()
        print(temp.keys())
        model_name = temp[model_name_key]
        results_processed[model_name] = {}
        for k in temp.keys():
            if data in ["CIFAR-10-C"]:
                # Means we have a dictionary with the topk accuracies per key
                if ("topk" in k) and (k not in ignore_keys):
                    for k2, v2 in temp[k].items():
                        # Keep only top1 accuracy, multiply by 100, show only 1 decimal digit
                        results_processed[model_name][k2] = "%.1f" % (temp[k][k2][0]*100)
            elif data in ["attacks_non_Lp", "attacks_Lp"]:
                # Ignore keys not corresponding to accuracies
                if ("bool" in k) or (k in ignore_keys):
                    pass
                else:
                    results_processed[model_name][k] = "%.1f" % (temp[k]*100)
            elif data in ["SVHN", "CIFAR100"]:
                # Format mean +/- std
                results_processed[model_name]["Accuracy"] = str("%.1f" % (temp["topk_accuracies_mean"][0]*100)) + u"\u00B1" + str("%.1f" % (temp["topk_accuracies_std"][0]*100))

    df_results = pandas.DataFrame.from_dict(results_processed)
    pandas.DataFrame.to_csv(df_results, RESULTS_PATH + "accuracies.csv")
    print(df_results, '\n\n')

parsed_args = argument_parser.parse_args()
dataset_to_print = parsed_args.data

if parsed_args.raw:
    print_raw_results(dataset_to_print)
else:
    results_to_csv(dataset_to_print)