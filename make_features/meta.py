import torch.nn as nn
import pandas as pd
import torch
from tqdm import tqdm
import json
from torch import load, device, cuda
from transformers import BertTokenizer, logging, BertModel, BertConfig
logging.set_verbosity_error()
from sys import argv

class Model(nn.Module):
    def __init__(self, num_classes:int, feature_size:int) -> None:
        super().__init__()
        self.config = BertConfig(max_position_embeddings=2048, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        self.bert = BertModel(self.config)
        self.features = nn.Linear(self.bert.config.hidden_size, feature_size)
        self.post_features = nn.Linear(feature_size, 200)
        self.output_layer = nn.Linear(200, num_classes)
        self.activation = nn.functional.tanh

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        out = self.features(encoded_input)
        features = self.activation(out)
        out = self.post_features(features)
        out = torch.nn.functional.relu(out)
        out = self.output_layer(out)
        out = torch.nn.functional.softmax(out, dim=1)
        return torch.cat((features, out), dim=1)

def integer_ReLu(x:torch.Tensor):
    x = nn.functional.relu(x) * 100
    return torch.round(x)
   
class Language_features_generator:
    def __init__(self, names:'list', pre_trained_weights:'str', feature_size:'int', num_classes:'int', probabilities_only:'bool'=False) -> None:
        super().__init__()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.model = Model(num_classes, feature_size)
        self.model.load_state_dict(load(pre_trained_weights))
        self.model = self.model.to(self.device)
        self.names = names
        self.model.eval()
        self.probabilities_only = probabilities_only

    def generate(self, tokenized_instance: 'dict') -> 'dict[str,float]':
        tokenized_instance = {k:tokenized_instance[k].to(self.device) for k in tokenized_instance.keys()}
        with torch.no_grad():
            model_output = self.model(tokenized_instance)[0]
        keys = list(tokenized_instance.keys())
        for k in keys:
            del tokenized_instance[k]
        if self.probabilities_only:
            return {self.names[i]: model_output["out"][i] for i in range(len(self.names))}
        else:
            out = {}
            for i in range(len(model_output)):
                out[f"feat{i}"] = round(float(model_output[i]), 3)
            return out

dataset_name = argv[1]
model_name = argv[2]
folds = argv[3]
feature_size = int(argv[4])
output_size = int(argv[5])
f = open(dataset_name)
dataset = json.load(f)
f.close()

cols = [d["combination"] for d in sorted(dataset[0]["all_times"], key= lambda x: x["combination"])]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True, model_max_length=2048)
for datapoint in dataset:
    datapoint["token"] = tokenizer(datapoint["instance_value_json"], truncation=True, return_tensors="pt")

fold_range = []
if ',' in folds:
    split_fold = folds.split(',')
    fold_range = [int(split_fold[0]), int(split_fold[1])]
else:
    fold_range = [int(folds)]

for i in range(*fold_range):
    new_features = []
    actual_name = model_name
    if '{i}' in model_name:
        split = model_name.split('{i}')
        actual_name = f'{split[0]}{i}{split[1]}'
    generator = Language_features_generator(cols, actual_name, feature_size, output_size)
    for datapoint in tqdm(dataset, desc=f"fold: {i}"):
        feature_gen = generator.generate(datapoint["token"])
        feature_gen["inst"] = datapoint["instance_name"]
        new_features.append(feature_gen)
    del generator.model
    del generator
    torch.cuda.empty_cache()
    new_features_df = pd.DataFrame(new_features)
    new_features_df.to_csv(f"features_{i}.csv", index=False)
