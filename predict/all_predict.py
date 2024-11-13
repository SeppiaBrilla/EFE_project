import argparse
import argcomplete
from time import time
import pandas as pd
import json
from helper import get_dataloader, is_competitive, positive_int, get_predictor

def get_features(instances, features) -> 'list[dict]':
    return [{
        "inst": inst[0], 
        "features": features[features["inst"] == inst[0]].to_numpy()[0][1:].tolist(), 
        "times": {t["combination"]: t["time"] for t in inst[1]}
    } for inst in instances]

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
    args["tune"] = parser.tune
    args["filter"] = parser.filter
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
parser.add_argument("-b","--base_folder", type=str, help="base folder")
parser.add_argument("--tune", default=False, help="If tune autofolio or not. Default = False", action='store_true')
argcomplete.autocomplete(parser)

def main():
    arguments = parser.parse_args()
    f = open(arguments.dataset)
    dataset = json.load(f)
    f.close()
    fold = arguments.split_fold
    original_features = pd.read_csv(arguments.features)
    save_folder = arguments.base_folder
    (x_train, _), (x_validation, _), (x_test, _) = get_dataloader(dataset, dataset, [fold])
    train_instances = [(x["instance_name"], x["all_times"]) for x in x_train]
    validation_instances = [(x["instance_name"], x["all_times"]) for x in x_validation]
    test_instances = [(x["instance_name"], x["all_times"]) for x in x_test]

    idx2comb = {idx:comb for idx, comb in enumerate(sorted([t["combination"] for t in x_train[0]["all_times"]]))}

    train_data = []
    for datapoint in x_train + x_validation:
        if original_features[original_features["inst"] == datapoint["instance_name"]].empty or original_features[original_features["inst"] == datapoint["instance_name"]].isna().any().any():
            continue
        train_data.append({
            "trues": [0 if is_competitive(datapoint["time"], t["time"]) else 1 for t in sorted(datapoint["all_times"], key=lambda x: x["combination"])],
            "inst": datapoint["instance_name"],
            "times": {t["combination"]:t["time"] for t in datapoint["all_times"]}
        })

    times = {}
    for datapoint in x_train + x_validation + x_test:
        times[datapoint["instance_name"]] = {t["combination"]:t["time"] for t in datapoint["all_times"]}

    opt_times = {comb["combination"]:0 for comb in dataset[0]["all_times"]}
    for datapoint in dataset:
        for t in datapoint["all_times"]:
            opt_times[t["combination"]] += t["time"]
    sb_key = min(opt_times.items(), key = lambda x: x[1])[0]

    args = build_kwargs(arguments, idx2comb, original_features)
    predictor = get_predictor(arguments.type, train_data, **args)
    total_time = 0
    sb_tot = 0
    train_features = [
        {"inst": inst[0], "features": original_features[original_features["inst"] == inst[0]].to_numpy()[0].tolist()} 
        for inst in train_instances 
        if not original_features[original_features["inst"] == inst[0]].empty
    ]

    for i in range(len(train_features)):
        train_features[i]["features"].pop(train_features[i]["features"].index(train_features[i]["inst"]))
        _ = [float(e) for e in train_features[i]["features"]]
    
    predictions = predictor.predict(train_features, filter=arguments.filter)
    assert len(predictions) ==  len(train_features)
    for inst in train_instances:
        if original_features[original_features["inst"] == inst[0]].empty:
            predictions.append({"chosen_option": sb_key, "inst": inst[0], "time": 0})

    f = open(f"{save_folder}/train_predictions_fold_{fold}", "w")
    json.dump(predictions, f)
    f.close()

    val_features = [
        {"inst": inst[0], "features": original_features[original_features["inst"] == inst[0]].to_numpy()[0].tolist()} 
        for inst in validation_instances
        if not original_features[original_features["inst"] == inst[0]].empty
    ]
    for i in range(len(val_features)):
        val_features[i]["features"].pop(val_features[i]["features"].index(val_features[i]["inst"]))
        _ = [float(e) for e in val_features[i]["features"]]
    predictions = predictor.predict(val_features, filter=arguments.filter)
    assert len(predictions) ==  len(val_features)
    for inst in validation_instances:
        if original_features[original_features["inst"] == inst[0]].empty:
            predictions.append({"chosen_option": sb_key, "inst": inst[0], "time": 0})

    f = open(f"{save_folder}/validation_predictions_fold_{fold}", "w")
    json.dump(predictions, f)
    f.close()

    test_features = [
        {"inst": inst[0], "features": original_features[original_features["inst"] == inst[0]].to_numpy()[0].tolist()} 
        for inst in test_instances
        if not original_features[original_features["inst"] == inst[0]].empty
    ]
    for i in range(len(test_features)):
        test_features[i]["features"].pop(test_features[i]["features"].index(test_features[i]["inst"]))
        _ = [float(e) for e in test_features[i]["features"]]
    predictions = predictor.predict(test_features, filter=arguments.filter)
    assert len(predictions) ==  len(test_features)
    for inst in test_instances:
        if original_features[original_features["inst"] == inst[0]].empty:
            predictions.append({"chosen_option": sb_key, "inst": inst[0], "time": 0})

    test_time = 0
    sb_test = 0
    for pred in predictions:
        sb_test += times[pred["inst"]][sb_key]
        test_time += times[pred["inst"]][pred["chosen_option"]]
    total_time += test_time
    sb_tot += sb_test
    f = open(f"{save_folder}/test_predictions_fold_{fold}", "w")
    json.dump(predictions, f)
    f.close()
    
    print(f"""
test: {test_time/sb_test:,.2f}
          """)

if __name__ == "__main__":
    main()
