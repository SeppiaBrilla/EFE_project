import argparse
import json
from time import time
from feature_generators.dnn_generator import Language_features_generator
from feature_generators.fzn2feat_generator import Fzn2feat_generator

def generate_dnn_features(args) -> "dict":
    if args.names is None:
        raise Exception("argument names is required with the dnn generation")
    if args.weights is None:
        raise Exception("argument weights is required with the dnn generation")
    generator = Language_features_generator(args.names.split(","), args.weights, args.probability_only)
    f = open(args.instance)
    instance = f.read()
    f.close()
    start_time = time()
    features = generator.generate(instance)
    end_time = time() - start_time
    if args.time:
        features = {"time": end_time, "features":features}
    return features

def generate_fzn2feat_features(args) -> "dict":
    if args.eprime is None:
        raise Exception("argument eprime is required with the fzn2feat generation")
    generator = Fzn2feat_generator(args.eprime)
    start_time = time()
    features = generator.generate(args.instance)
    end_time = time() - start_time
    if args.time:
        features = {"time": end_time, "features":features}
    return features

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", choices=["dnn", "fzn2feat"], help="The type of features to get. Default = dnn", default="dnn")
parser.add_argument("-i", "--instance", type=str, help="The file containing the instance to use to generate the features", required=True)
parser.add_argument("-n", "--names", type=str, help="A comma separated list of names to use as names for the probability features with the dnn features")
parser.add_argument("-p", "--probability-only", help="If the features should be only the predicted probabilities (dnn only). Default = False", 
                    default=False, action='store_true')
parser.add_argument("-w", "--weights", type=str, help="The weights to load for the dnn")
parser.add_argument("-e", "--eprime", type=str, help="The eprime file to use to generate the features (fzn2feat only)")
parser.add_argument("-o", "--output", choices=["json", "csv"], help="The output format. Default= csv", default="csv")
parser.add_argument("--time", help="If the program should also report the time taken to generate the features. Default = False", 
                    default=False, action='store_true')

def main():
    arguments = parser.parse_args()
    features = {}
    if arguments.type == "dnn":
        features = generate_dnn_features(arguments)
    elif arguments.type == "fzn2feat":
        features = generate_fzn2feat_features(arguments)

    if arguments.output == "json":
        print(json.dumps(features))
    elif arguments.output == "csv":
        if "time" in features:
            keys = ",".join(["time"] + list(features["features"].keys()))
            values = ",".join([str(features["time"])] + [str(features["features"][k]) for k in features["features"].keys()])
            output = f"{keys}\n{values}"
        else:
            keys = ",".join(list(features.keys()))
            values = ",".join([str(features[k]) for k in features.keys()])
            output = f"{keys}\n{values}"
        print(output)

if __name__ == "__main__":
    main()
