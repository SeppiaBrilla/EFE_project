import argparse
import joblib
from time import time
import os
import pandas as pd
import shutil
import json
from helper import is_competitive
from predictor.base_predictor import Predictor
from predictor.autofolio_predictor import Autofolio_predictor, Autofolio_initializer
from predictor.clustering_predictor import Kmeans_predictor, kmeans_initializer
from predictor.order_predictor import Static_ordering_predictor, Static_ordering_initializer
from predictor.order_metrics import Metrics_predictor, Metrics_initializer

CONFIG_NAME = "config"

def load(name) -> Predictor:
    f = open(os.path.join(name, CONFIG_NAME))
    config = json.load(f)
    if config["predictor_type"] == "autofolio":
        initializer = Autofolio_initializer(os.path.join(name, Autofolio_predictor.MODEL_NAME), config["max_threads"])
        return Autofolio_predictor.from_pretrained(initializer)
    elif config["predictor_type"] == "static":
        initializer = Static_ordering_initializer(config["order"], config["idx2comb"])
        return Static_ordering_predictor.from_pretrained(initializer)
    elif config["predictor_type"] == "metric":
        initializer = Metrics_initializer(config["order"], config["idx2comb"])
        return Metrics_predictor.from_pretrained(initializer)
    elif config["predictor_type"] == "kmeans":
        initializer = kmeans_initializer(os.path.join(name, Kmeans_predictor.MODEL_NAME), config["order"], config["idx2comb"])
        return Kmeans_predictor.from_pretrained(initializer)
    else:
        raise Exception(f"predictor_type {config['predictor_type']} unrecognised")

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

def train(arguments):
    times = pd.read_csv(arguments.times)
    features = pd.read_csv(arguments.features)

    combinations = list(times.columns)
    if not "inst" in combinations:
        raise Exception("The time file does not include a instance column.")

    combinations.pop(combinations.index("inst"))
    idx2comb = {idx:comb for idx, comb in enumerate(combinations)}
    predictor_type = arguments.type

    if arguments.type in ["static", "metric"] and not len(times.columns) == len(features.columns):
        raise Exception(f"predictor of type {arguments.type} must filter out the options and to do so, the features must be the same as the number of options")
    train_data = []

    for i in range(len(features)):
        inst = features[i]["inst"]
        true_times = times[times["inst"] == inst].iloc[0][combinations]
        vb = min(true_times)

        train_data.append({
            "trues": [0 if is_competitive(vb, t) else 1 for t in true_times],
            "inst": inst,
            "times": true_times,
        })

    start_time = time()
    data_to_save = {"idx2comb":idx2comb, "predictor_type":predictor_type}
    if predictor_type == "static":
        if arguments.ordering is None:
            raise Exception(f"predictor_type {predictor_type} needs an ordering type. ordering_type cannot be None")
        predictor = Static_ordering_predictor(idx2comb=idx2comb, training_data=train_data, ordering_type=arguments.ordering)
        data_to_save["order"] = predictor.order
    elif predictor_type == "kmeans":
        predictor = Kmeans_predictor(training_data=train_data, idx2comb=idx2comb, features=features,filter=arguments.filter) 
        data_to_save["order"] = predictor.order
    elif predictor_type == "autofolio":
        predictor = Autofolio_predictor(training_data=train_data, features=features, max_threads=arguments.max_threads)
        data_to_save["max_threads"] = predictor.max_threads
    elif predictor_type == "metric":
        if arguments.metrics_type is None:
            raise Exception(f"predictor_type {predictor_type} needs a metric type. metrics_type cannot be None")
        predictor = Metrics_predictor(training_data=train_data, idx2comb=idx2comb, features=features, metrics_type=arguments.metrics_type)
        data_to_save["order"] = predictor.order
    else:
        raise Exception(f"predictor_type {predictor_type} unrecognised")

    if arguments.time:
        print(f"The predictor took {time() - start_time:,.2f} seconds to create")

    if not os.path.isdir(arguments.name):
        os.mkdir(arguments.name)
    f = open(os.path.join(arguments.name, CONFIG_NAME), "w")
    json.dump(data_to_save, f)
    if predictor_type == "autofolio":
        shutil.copy(predictor.model, os.path.join(arguments.name, Autofolio_predictor.MODEL_NAME))
    if predictor_type == "kmeans":
        joblib.dump(predictor.clustering_model, os.path.join(arguments.name, Kmeans_predictor.MODEL_NAME))

def predict(args):
    predictor = load(args.name)
    features = args.features
    if "," in features:
        features = [float(v) for v in features.split(",")]
        start_time = time()
        output = {}
        output["chosen_option"] = predictor.predict(features)
        if args.time:
            output["prediction_time"] = time() - start_time
        if args.output == "text":
            output = "\n".join([f"{key}:    {output[key]}" for key in output.keys()])
        elif args.output == "json":
            output = json.dumps(output)
        elif args.output == "csv":
            keys = list(output.keys())
            output = f"{'.'.join(keys)}\n{','.join([str(output[k]) for k in keys])}"
        print(output)

    elif os.path.exists(features):
        df = pd.read_csv(features)
        instances = df["inst"].to_list()
        features = [{"inst": inst, "features": df[df["inst"] == inst].to_numpy()[0].tolist()[1:]} for inst in instances]
        start_time = time()
        predictions = predictor.predict(features)
        final_time = time() - start_time
        output = {}
        output["predictions"] = predictions
        if args.time:
            output["prediction_time"] = final_time
        if args.output == "text":
            output = "predictions:"
            for prediction in predictions:
                output += f"\n\t- {prediction['inst']}:  {prediction['chosen_option']}"
            output += f"\nprediction time:  {final_time}"
        elif args.output == "json":
            output = json.dumps(output)
        elif args.output == "csv":
            output = "inst,chosen_option,total_time"
            for prediction in predictions:
                output += f"{prediction['inst']},{prediction['chosen_option']},{final_time}"
        print(output)
    

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", choices=["train", "predict"], help="mode for the script. train: train a classifier, predict (default): predict using a classifier", default="predict")
parser.add_argument("--type", choices=["static", "kmeans", "autofolio", "metric"], help="the heuristic to use to make a choiche", required=False)
parser.add_argument("-t", "--times", type=str, help="The times of each option to use in the training process", required=False)
parser.add_argument("-o", "--ordering", choices=["single_best", "wins"], help="the heuristic to use to make a choiche in the static ordering")
parser.add_argument("--filter", default=False, help="Whether if the model should use the feature to pre-filter the options or not. Default = False", action='store_true')
parser.add_argument("-f", "--features", type=str, help="The features to use with the heuristic. It must be either a csv file or a comma separated list of values to use as features", required=True)
parser.add_argument("--metrics_type", type=str, help="The metric to maximise in the metric ordering", choices=["recall", "accuracy", "precision", "f1"], required=False)
parser.add_argument("--output", choices=["text", "json", "csv"], help="The output format when predicting. Default: text", default="text")
parser.add_argument("--max_threads", help="The number of threads to use during the prediction with autofolio", type=int, default=12)
parser.add_argument("--time", default=False, help="Whether the script shoud print the time required to get the predictions or not. Default = False", action='store_true')
parser.add_argument("-n", "--name", help="File name to use for the predictor", type=str, required=True)

def main():
    args = parser.parse_args()
    if args.mode == "predict":
        predict(args)
    elif args.mode == "train":
        train(args)

if __name__ == "__main__":
    main()
