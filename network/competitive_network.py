from typing import Callable, Tuple
import torch.nn as nn
import argcomplete
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch
from json import loads, dump
from helper import dict_lists_to_list_of_dicts, get_dataloader
from sklearn.metrics import accuracy_score, f1_score
from transformers import  BertModel, BertConfig, BertTokenizer
from sys import stdout

BERT_TYPE = "bert-base-uncased"

def integer_ReLu(x:torch.Tensor):
    x = nn.functional.relu(x) * 100
    return torch.round(x)

def integer_Sigmoid(x:torch.Tensor):
    x = nn.functional.sigmoid(x)
    return torch.round(x)

def integer_Tanh(x:torch.Tensor):
    x = nn.functional.tanh(x)
    return torch.round(x)

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

def is_competitive(vb, option):
    return (option < 10 or vb * 2 >= option) and option < 3600

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
    del data

def train_one_epoch(model:nn.Module, train_dataloader:DataLoader, optimizer:torch.optim.Optimizer, loss:Callable, device:torch.device):
    n_batch = len(train_dataloader)

    for i, data in enumerate(train_dataloader):
        inputs, labels = data

        inputs = to(inputs, device)
        labels = to(labels, device)

        optimizer.zero_grad()

        outputs = model(inputs)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()

        remove(inputs)
        remove(labels)

        stdout.write(f"\r{i}/{n_batch}  {l.item():.2f}")
        stdout.flush()

def compute_total_loss_and_predictions(model:nn.Module, 
                                       dataloader:DataLoader, 
                                       loss:Callable, 
                                       device:torch.device):
    total_loss = 0
    total_elements = len(dataloader)
    total_predictions = []
    total_labels = []
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, labels = data

            inputs = to(inputs, device)
            labels = to(labels, device)

            outputs = model(inputs)
            total_loss += loss(outputs, labels).item()
            predictions = nn.functional.sigmoid(outputs).round()
            total_predictions += predictions.tolist()
            total_labels += labels.tolist()

            remove(inputs)
            remove(labels)

    return total_loss / total_elements, (total_predictions, total_labels)

def train(model:nn.Module, 
          train_dataset:DataLoader, 
          validation_dataset:DataLoader, 
          optimizer:torch.optim.Optimizer, 
          loss:Callable,
          epochs: int,
          device:torch.device,
          hyperparam:dict) -> Tuple[nn.Module, dict]:

    model = model.to(device)
    data = {"train": {"loss":[], "accuracy":[], "f1":[]}, "validation": {"loss":[], "accuracy":[], "f1":[]}}

    best_model = CompetitiveModel(**hyperparam)
    best_loss = np.inf

    for epoch in range(epochs):
        model.train()
        train_one_epoch(model, train_dataset, optimizer, loss, device)
        model.eval()
        train_loss, (predicted_labels, true_labels) = compute_total_loss_and_predictions(model, train_dataset, loss, device)
        predicted_labels, true_labels = np.array(predicted_labels), np.array(true_labels)
        train_accuracy = accuracy_score(true_labels.reshape(-1), predicted_labels.reshape(-1))
        train_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=1)
        validation_loss, (true_labels, predicted_labels) = compute_total_loss_and_predictions(model, validation_dataset, loss, device)
        predicted_labels, true_labels = np.array(predicted_labels), np.array(true_labels)
        validation_accuracy = accuracy_score(true_labels.reshape(-1), predicted_labels.reshape(-1))
        validation_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=1)

        data["train"]["loss"].append(train_loss)
        data["train"]["accuracy"].append(train_accuracy)
        data["train"]["f1"].append(train_f1)

        data["validation"]["loss"].append(validation_loss)
        data["validation"]["accuracy"].append(validation_accuracy)
        data["validation"]["f1"].append(validation_f1)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model.load_state_dict(model.state_dict())

        # saver(model)
        out_str = f"""epoch: {epoch + 1}/{epochs} train: (loss: {train_loss:.3f}, accuracy: {train_accuracy:.2f}, f1: {train_f1:.2f}) validation: (loss: {validation_loss:.3f}, accuracy: {validation_accuracy:.2f}, f1: {validation_f1:.2f})|"""
        out_str += "\n" + "-" * len(out_str)

        stdout.write("\r" + " " * len(out_str) + "\r")
        stdout.flush()
        stdout.write(out_str + "\n")
    
    return best_model, data


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--save", required=True)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--pre_trained", required=False)
parser.add_argument("--history", required=False)
parser.add_argument("--features_size", required=False, type=int)
parser.add_argument("--limit", required=False, type=int, default=0)
parser.add_argument("--batch-size", required=False, type=int, default=4)

argcomplete.autocomplete(parser)
def main():

    arguments = parser.parse_args()
    dataset = arguments.dataset
    pretrained_weights = arguments.pre_trained
    epochs = arguments.epochs
    learning_rate = arguments.learning_rate
    save_weights_file = arguments.save
    fold = arguments.fold
    history_file = arguments.history
    feature_size = arguments.features_size

    batch_size = arguments.batch_size

    f = open(dataset)
    data = loads(f.read())
    f.close()

    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    instances_and_model = [d["instance_value_json"] for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = []

    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
    for datapoint in data:
        y_datapoint = sorted(datapoint["all_times"], key= lambda x: x["combination"])
        datapoint["all_times"] = y_datapoint
        ordered_times = [d["time"] for d in datapoint["all_times"]]
        ordered_times = sorted(ordered_times)
        vb = min([d["time"] for d in y_datapoint])
        competitivness = [1. if is_competitive(vb, d["time"]) else 0. for d in y_datapoint if d["combination"] in combinations]
        y.append(torch.Tensor(competitivness))

    train_dataloader, validation_dataloader, _ = get_dataloader(x, y, batch_size, [fold])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    length = len(combinations)

    model = CompetitiveModel(feature_size, length)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_elements = len(y)
    total_competitives = [sum([yi[i] for yi in y]) for i in range(length)]
    weights = [1 + (1 - (total_competitives[i]/total_elements)) for i in range(length)]
    weights = torch.tensor(weights, device=device)
    def loss(pred, true):
        logits = nn.functional.binary_cross_entropy_with_logits(pred, true, reduction="none")
        logits = logits * weights

        return torch.mean(logits)

    model, train_data = train(model, train_dataloader, validation_dataloader, optimizer, loss, epochs, device, {"feature_size": feature_size, "output_size": length})
    torch.save(model.state_dict(), f"{save_weights_file}_final")

    f = open(history_file, "w")
    dump(train_data, f)
    f.close()
    
main()
