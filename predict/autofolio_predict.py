import argparse
import argcomplete
from time import time
import pandas as pd
import json
from helper import get_dataloader, is_competitive, positive_int
from predictor.autofolio_predictor import Autofolio_initializer, Autofolio_predictor

def get_features(instances, features) -> 'list[dict]':
    return [{
        "inst": inst[0], 
        "features": features[features["inst"] == inst[0]].to_numpy()[0][1:].tolist(), 
        "times": {t["combination"]: t["time"] for t in inst[1]}
    } for inst in instances]

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--features", type=str, help="The features to use (in csv format) with the heuristic", required=True)
parser.add_argument("-d", "--dataset", type=str, help="The dataset to use (in json format)", required=True)
parser.add_argument("-s", "--split-fold", type=positive_int, help="The fold to use to split the dataset", required=True)
parser.add_argument("-p", "--pre-trained-model", type=str, help="The path to a pre-trained Autofolio model", required=False)
parser.add_argument("-b","--base_folder", type=str, help="base folder")
argcomplete.autocomplete(parser)

def main():
    arguments = parser.parse_args()
    f = open(arguments.dataset)
    dataset = json.load(f)
    f.close()
    fold = arguments.split_fold
    original_features = pd.read_csv(arguments.features)
    save_folder = arguments.base_folder
    model_name = arguments.pre_trained_model

    (x_train, _), (x_validation, _), (x_test, _) = get_dataloader(dataset, dataset, [fold])
    train_instances = [(x["instance_name"], x["all_times"]) for x in x_train]
    validation_instances = [(x["instance_name"], x["all_times"]) for x in x_validation]
    test_instances = [(x["instance_name"], x["all_times"]) for x in x_test]

    train_features = get_features(train_instances, original_features)
    test_features = get_features(test_instances, original_features)


    train_data = []
    for datapoint in x_train + x_validation:
        train_data.append({
            "trues": [0 if is_competitive(datapoint["time"], t["time"]) else 1 for t in sorted(datapoint["all_times"], key=lambda x: x["combination"])],
            "inst": datapoint["instance_name"],
            "times": {t["combination"]:t["time"] for t in datapoint["all_times"]}
        })

    times = {}
    for datapoint in x_train + x_validation + x_test:
        times[datapoint["instance_name"]] = {t["combination"]:t["time"] for t in datapoint["all_times"]}

    pre_trained = Autofolio_initializer(model_name, 5)
    predictor = Autofolio_predictor.from_pretrained(pre_trained)
    total_time = 0
    sb_tot = 0
    train_features = [{"inst": inst[0], "features": original_features[original_features["inst"] == inst[0]].to_numpy()[0].tolist()} for inst in train_instances]

    for i in range(len(train_features)):
        train_features[i]["features"].pop(train_features[i]["features"].index(train_features[i]["inst"]))
        _ = [float(e) for e in train_features[i]["features"]]
    
    predictions = predictor.predict(train_features)
    assert len(predictions) ==  len(train_features)

    f = open(f"{save_folder}/train_predictions_fold_{fold}", "w")
    json.dump(predictions, f)
    f.close()

    val_features = [{"inst": inst[0], "features": original_features[original_features["inst"] == inst[0]].to_numpy()[0].tolist()} for inst in validation_instances]
    for i in range(len(val_features)):
        val_features[i]["features"].pop(val_features[i]["features"].index(val_features[i]["inst"]))
        _ = [float(e) for e in val_features[i]["features"]]
    predictions = predictor.predict(val_features)
    assert len(predictions) ==  len(val_features)

    f = open(f"{save_folder}/validation_predictions_fold_{fold}", "w")
    json.dump(predictions, f)
    f.close()

    test_features = [{"inst": inst[0], "features": original_features[original_features["inst"] == inst[0]].to_numpy()[0].tolist()} for inst in test_instances]
    for i in range(len(test_features)):
        test_features[i]["features"].pop(test_features[i]["features"].index(test_features[i]["inst"]))
        _ = [float(e) for e in test_features[i]["features"]]
    predictions = predictor.predict(test_features)
    assert len(predictions) ==  len(test_features)

    opt_times = {comb["combination"]:0 for comb in dataset[0]["all_times"]}
    for datapoint in dataset:
        for t in datapoint["all_times"]:
            opt_times[t["combination"]] += t["time"]
    sb_key = min(opt_times.items(), key = lambda x: x[1])[0]
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
