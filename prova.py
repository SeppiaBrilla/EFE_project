import json, os, subprocess
from make_features.feature_generators.dnn_generator import Language_features_generator
from make_features.feature_generators.fzn2feat_generator import Fzn2feat_generator
import tqdm
import pandas as pd
from time import time
f = open("data/datasets/dataset_SocialGolfers-2024-05-16.json")
dataset = json.load(f)
f.close()
generator = Fzn2feat_generator("../EssenceCatalog-runs/problems/csplib-prob010-SocialGolfers/conjure-mode/portfolio4/01_compact.eprime")
features = []
for datapoint in tqdm.tqdm(dataset):
    start = time()
    res = generator.generate(f"../{datapoint['instance_name']}")
    end = time()
    features.append({"inst":datapoint["instance_name"], "features": res, "time": end - start})

f = open(f"fzn2feat_social_golfers.json", "w")
json.dump(features, f)
f.close()