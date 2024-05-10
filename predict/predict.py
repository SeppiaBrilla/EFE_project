import argparse
from time import time
import pandas as pd
import os
import json
from helper import get_dataloader, is_competitive, get_sb_vb, positive_int, get_predictor, pad
from predict.predictor.autofolio_predictor import Autofolio_predictor
from predict.predictor.clustering_predictor import Clustering_predictor

def get_features(instances, features) -> 'list[dict]':
    return [{
        "inst": inst[0], 
        "features": features[features["inst"] == inst[0]].to_numpy()[0][1:].tolist(), 
        "times": {t["combination"]: t["time"] for t in inst[1]}
    } for inst in instances]

def dnn_filtering(dataset:'list[dict]') -> 'list[dict]':
    filtered_dataset = []
    for datapoint in dataset:
        filtered_dataset.append({
            "features": datapoint["features"],
            "times": datapoint["times"],
            "inst": datapoint["inst"]
        })
    return filtered_dataset

def build_kwargs(parser, idx2comb, features):
    args = {}
    args["fold"] = parser.split_fold
    args["idx2comb"] = idx2comb
    args["features"] = features
    if not parser.ordering is None:
        args["ordering_type"] = parser.ordering
    if not parser.hyperparameters is None:
        args["hyperparameters"] = parser.hyperparameters
    if not parser.max_threads is None:
        args["max_threads"] = parser.max_threads
    if not parser.pre_trained_model is None:
        args["pre_trained_model"] = parser.pre_trained_model
    if not parser.metrics_type is None:
        args["metrics_type"] = parser.metrics_type
    return args

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", choices=["static", "kmeans", "autofolio", "metric"], help="the heuristic to use to make a choiche", required=True)
parser.add_argument("-o", "--ordering", choices=["single_best", "wins"], help="the heuristic to use to make a choiche in the static ordering")
parser.add_argument("--filter", default=False, help="Whether if the model should use the feature to pre-filter the options or not. Default = False", action='store_true')
parser.add_argument("-f", "--features", type=str, help="The features to use (in csv format) with the heuristic", required=True)
parser.add_argument("-d", "--dataset", type=str, help="The dataset to use (in json format)", required=True)
parser.add_argument("-s", "--split-fold", type=positive_int, help="The fold to use to split the dataset", required=True)
parser.add_argument("--hyperparameters", type=str, help="A json file containing the hyperparameters to use with the kmeans clustering.", required=False)
parser.add_argument("--max_threads", type=int, help="The maximum number of threads to use with Autofolio. Default is 12", required=False)
parser.add_argument("--pre_trained_model", type=str, help="The path to a pre-trained Autofolio model", required=False)
parser.add_argument("-m", "--metrics_type", type=str, help="The metric to maximise in the metric ordering", choices=["recall", "accuracy", "precision", "f1"], required=False)
parser.add_argument("--time", default=False, help="Whether the script shoud print the time required to get the predictions or not. Default = False", action='store_true')
parser.add_argument("--save", help="save file where to store the model/hyperparameters. Used only with kmeans and autofolio", type=str)

def main():
    arguments = parser.parse_args()
    f = open(arguments.dataset)
    dataset = json.load(f)
    f.close()
    fold = arguments.split_fold
    features = pd.read_csv(arguments.features)
    (x_train, _), (x_validation, _), (x_test, _) = get_dataloader(dataset, dataset, [fold])
    train_instances = [(x["instance_name"], x["all_times"]) for x in x_train]
    validation_instances = [(x["instance_name"], x["all_times"]) for x in x_validation]
    test_instances = [(x["instance_name"], x["all_times"]) for x in x_test]

    train_features = get_features(train_instances, features)
    validation_features = get_features(validation_instances, features)
    test_features = get_features(test_instances, features)

    train_filtered_options = dnn_filtering(train_features)
    validation_filtered_options = dnn_filtering(validation_features)
    test_filtered_options = dnn_filtering(test_features)

    idx2comb = {idx:comb for idx, comb in enumerate(sorted([t["combination"] for t in x_train[0]["all_times"]]))}

    train_data = []
    for datapoint in x_train + x_validation:
        train_data.append({
            "trues": [0 if is_competitive(datapoint["time"], t["time"]) else 1 for t in sorted(datapoint["all_times"], key=lambda x: x["combination"])],
            "inst": datapoint["instance_name"],
            "times": {t["combination"]:t["time"] for t in datapoint["all_times"]}
        })

    args = build_kwargs(arguments, idx2comb, features)
    start_time = time()
    predictor = get_predictor(arguments.type, train_data, **args)
    if arguments.time:
        print(f"The predictor took {time() - start_time:,.2f} seconds to create")

    if not arguments.save is None:
        if isinstance(predictor, Clustering_predictor):
            f = open(arguments.save, "w")
            json.dump(predictor.clustering_parameters, f)
        if isinstance(predictor, Autofolio_predictor):
            os.rename(predictor.model, arguments.save)
    (sb_train, vb_train), \
    (sb_val, vb_val), \
    (sb_test, vb_test) = get_sb_vb(x_train, x_validation, x_test)

    mult = 85
    
    start_time = time()
    _, total_time = predictor.predict(train_filtered_options, filter=arguments.filter)
    print("-"*mult)
    if arguments.time:
        print(f"The predictor took {time() - start_time:,.2f} seconds to predict the training set ({len(train_filtered_options)} elements).")
    sb_perc, vb_perc = total_time / sb_train, total_time / vb_train
    vb_train, total_time, sb_train = pad(f"{vb_train:,.2f}", f"{total_time:,.2f}", f"{sb_train:,.2f}")
    print(f"""The final results for the train set are: 
    virtual best:   {vb_train}
    predictor:      {total_time} ({sb_perc:.2f} of single best and {vb_perc:.2f} of virtual best)
    single best:    {sb_train}""")

    start_time = time()
    _, total_time = predictor.predict(validation_filtered_options, filter=arguments.filter)
    print("-"*mult)
    if arguments.time:
        print(f"The predictor took {time() - start_time:,.2f} seconds to predict the validation set ({len(validation_filtered_options)} elements).")
    sb_perc, vb_perc = total_time / sb_val, total_time / vb_val
    vb_val, total_time, sb_val = pad(f"{vb_val:,.2f}", f"{total_time:,.2f}", f"{sb_val:,.2f}")
    print(f"""The final results for the validation set are: 
    virtual best:   {vb_val}
    predictor:      {total_time} ({sb_perc:.2f} of single best and {vb_perc:.2f} of virtual best)
    single best:    {sb_val}""")

    start_time = time()
    _, total_time = predictor.predict(test_filtered_options, filter=arguments.filter)
    print("-"*mult)
    if arguments.time:
        print(f"The predictor took {time() - start_time:,.2f} seconds to predict the test set ({len(test_filtered_options)} elements).")
    sb_perc, vb_perc = total_time / sb_test, total_time / vb_test
    vb_test, total_time, sb_test = pad(f"{vb_test:,.2f}", f"{total_time:,.2f}", f"{sb_test:,.2f}")
    print(f"""The final results on the test set are: 
    virtual best:   {vb_test}
    predictor:      {total_time} ({sb_perc:.2f} of single best and {vb_perc:.2f} of virtual best)
    single best:    {sb_test}""")

if __name__ == "__main__":
    main()
