import torch
import argcomplete
import torch.nn as nn
from json import loads, dump
from tqdm import tqdm
import argparse
from time import time
from transformers import BertTokenizer, BertConfig, BertModel

class CompetitiveModel(nn.Module):
    def __init__(self, feature_size, output_size) -> None:
        super().__init__()
        self.config = BertConfig(max_position_embeddings=2048, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        self.bert = BertModel(self.config)
        self.features = nn.Linear(self.bert.config.hidden_size,feature_size)
        self.dropout = nn.Dropout(.3)
        self.post_features = nn.Linear(feature_size, 200)
        self.output_layer = nn.Linear(200, output_size)
        self.activation = nn.functional.tanh

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        out = self.features(encoded_input)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.post_features(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        return self.output_layer(out)
    
def split(x, test_buckets = []):
    BUCKETS = 10

    N_ELEMENTS = len(x)

    BUCKET_SIZE = N_ELEMENTS // BUCKETS

    x_local = x.copy()
    x_test = []

    for bucket in test_buckets:
        idx = bucket * BUCKET_SIZE
        for _ in range(BUCKET_SIZE):
            x_test.append(x_local.pop(idx))

    train_elements = (len(x_local) // 10) * 9
    x_train = x_local[:train_elements]

    x_validation = x_local[train_elements:]

    return x_train, x_validation, x_test

def to(data, device):
  if isinstance(data, dict):
    return {key: to(data[key], device) for key in data.keys()}
  elif isinstance(data, list):
    return [to(d, device) for d in data]
  elif isinstance(data, tuple):
    return tuple([to(d, device) for d in data])
  else:
    return data.to(device)
  
def remove(data):
  if isinstance(data, dict):
    for key in data.keys():
      remove(data[key])
  elif isinstance(data, list) or isinstance(data, tuple):
    for d in data:
      remove(d)
  else:
    del data


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-f", "--fold", type=int, required=True)
parser.add_argument("-b", "--base-folder", type=str, required=True)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("--features", type=int, required=True)
argcomplete.autocomplete(parser)

def main():

    args = parser.parse_args()
    dataset = args.dataset
    fold = args.fold
    base_folder = args.base_folder
    model_name = args.model
    BERT_TYPE = "bert-base-uncased"
    f = open(dataset)
    data = loads(f.read())
    f.close()
    train_instances, validation_instances, test_instances = split(data, [fold])
    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    length = len(train_instances[0]["all_times"])
    model =  CompetitiveModel(100, length)
    model.load_state_dict(torch.load(model_name))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    options = [t["combination"] for t in train_instances[0]["all_times"]]
    options = sorted(options)

    train_predictions = []
    for instance in tqdm(train_instances):
        t = time()
        x = tokenizer(instance["instance_value_json"], truncation=True, return_tensors = 'pt')
        x = to(x, device)
        y_pred = model(x)
        t = time() - t
        remove(x)
        y_pred = int(torch.argmax(y_pred))
        opt = options[y_pred]
        train_predictions.append({"chosen_option": opt, "inst": instance["instance_name"], "time": t})
    f = open(f'{base_folder}/train_predictions_fold_{fold}', 'w')
    dump(train_predictions, f)
    f.close()

    validation_predictions = []
    for instance in tqdm(validation_instances):
        t = time()
        x = tokenizer(instance["instance_value_json"], truncation=True, return_tensors = 'pt')
        x = to(x, device)
        y_pred = model(x)
        t = time() - t
        remove(x)
        y_pred = int(torch.argmax(y_pred))
        opt = options[y_pred]
        validation_predictions.append({"chosen_option": opt, "inst": instance["instance_name"], "time": t})
    f = open(f'{base_folder}/validation_predictions_fold_{fold}', 'w')
    dump(validation_predictions, f)
    f.close()


    opt_times = {comb["combination"]:0 for comb in data[0]["all_times"]}
    for datapoint in data:
        for t in datapoint["all_times"]:
            opt_times[t["combination"]] += t["time"]
    sb_key = min(opt_times.items(), key = lambda x: x[1])[0]

    pred_time = 0
    sb = 0
    test_predictions = []
    for instance in tqdm(test_instances):
        t = time()
        x = tokenizer(instance["instance_value_json"], truncation=True, return_tensors = 'pt')
        x = to(x, device)
        y_pred = model(x)
        t = time() - t
        remove(x)
        y_pred = int(torch.argmax(y_pred))
        opt = options[y_pred]
        current_times = {t['combination']:t['time'] for t in instance['all_times']}
        sb += current_times[sb_key]
        test_predictions.append({"chosen_option": opt, "inst": instance["instance_name"], "time": t})
        pred_time += current_times[opt]

    f = open(f'{base_folder}/test_predictions_fold_{fold}', 'w')
    dump(test_predictions, f)
    f.close()
    print(f"""
    pred/sb = {pred_time/sb:.2f}
""")

main()
