import torch
from json import loads
import numpy as np
from torch.utils.data import DataLoader
from helper import dict_lists_to_list_of_dicts, get_dataloader, remove_comments, save_predictions, get_time_matrix
from old_models import Timeout_and_selection_model, get_tokenizer
from sys import argv
 
class Evaluate_time:

    def __init__(self, len_train, len_val, len_test, min_train, min_validation, min_test, sb_train, sb_validation, sb_test, test_dataloader, train_dataloader, times_matrix) -> None:
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test
        self.vb_train = min_train
        self.vb_validation = min_validation
        self.vb_test = min_test
        self.sb_train = sb_train
        self.sb_validation = sb_validation
        self.sb_test = sb_test
        self.test_dataloader = test_dataloader
        self.times_matrix = times_matrix
        self.train_dataloader = train_dataloader 
        
    def __call__(self, model, loaders, device, output_extraction_function):
        model.eval()
        train_prediction = model.predict(self.train_dataloader, output_extraction_function, device)
        validation_prediction = model.predict(loaders["validation"], output_extraction_function, device)
        test_prediction = model.predict(self.test_dataloader, output_extraction_function, device)
        pred_train = round(sum([self.times_matrix[i, train_prediction[i]] for i in range(self.len_train)]), 2)
        pred_val = round(sum([self.times_matrix[self.len_train + i, validation_prediction[i]] for i in range(self.len_val)]), 2)
        pred_test = round(sum([self.times_matrix[self.len_train + self.len_val + i, test_prediction[i]] for i in range(self.len_test)]), 2)

        print("vb   pred    sb time")
        print(f"train set:\n{self.vb_train}     {pred_train}      {self.sb_train}")
        print(f"validation set:\n{self.vb_validation}     {pred_val}     {self.sb_validation}")
        print(f"test set:\n{self.vb_test}     {pred_test}     {self.sb_test}")
        print(f"\n{'-'*100}\n")



def main():
    
    if argv[1] == '--help':
        print("network.py dataset batch_size bert_type epochs learning_rate history_file prediction_file save_weights_file pretrained_weights")
        print("bert types: \n[1]FacebookAI/roberta-base \n[2]bert-base-uncased \n[3]allenai/longformer-base-4096")
        return
    
    dataset, batch_size, bert_type, prediction_file, pretrained_weights = argv[1], int(argv[2]), argv[3], argv[4], argv[5]
    
    if bert_type == "1":
        bert_type = "FacebookAI/roberta-base"
    elif bert_type == "2":
        bert_type = "bert-base-uncased"
    elif bert_type == "3":
        bert_type = "allenai/longformer-base-4096"

    f = open(dataset)
    data = loads(f.read())
    f.close()
    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
    length = len(combinations)
    model = Timeout_and_selection_model(bert_type, length, dropout=.3)
    model.load_state_dict(torch.load(pretrained_weights))


    tokenizer = get_tokenizer(bert_type)
    instances_and_model = [remove_comments(d["instance_value"]) for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = []

    for datapoint in data:
        y_datpoint = sorted(datapoint["all_times"], key= lambda x: x["combination"])
        datapoint["all_times"] = y_datpoint
        timeouts = [0 if d["time"] < 3600 else 1 for d in y_datpoint]
        algorithm_selection = torch.zeros(len(combinations))
        algorithm_selection[combinations.index(datapoint["combination"])] = 1
        y.append({
            "timeouts":torch.Tensor(timeouts),
            "algorithm_selection":algorithm_selection,
            "times": [d["time"] for d in y_datpoint]
        })
        
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, batch_size, [9])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    all_times = [datapoint["all_times"] for datapoint in data]
    times = get_time_matrix((len(all_times), len(combinations)), all_times)

    len_train = len(train_dataloader.dataset)
    len_val = len(validation_dataloader.dataset)
    len_test = len(test_dataloader.dataset)
    sb_train = min([np.sum(times[:len_train, i]) for i in range(len(combinations))])
    sb_val = min([np.sum(times[len_train:len_train + len_val,i]) for i in range(len(combinations))])
    sb_test = min([np.sum(times[len_train + len_val:,i]) for i in range(len(combinations))])

    min_train = np.sum([min(times[i, :]) for i in range(len_train)])
    min_val = np.sum([min(times[len_train + i, :]) for i in range(len_val)])
    min_test = np.sum([min(times[len_train + len_val + i, :]) for i in range(len_test)])
   
    in_between = Evaluate_time(len_train, len_val, len_test, 
                round(min_train, 2), round(min_val, 2), round(min_test, 2), 
                round(sb_train, 2), round(sb_val, 2), round(sb_test, 2), 
                test_dataloader, DataLoader(train_dataloader.dataset, shuffle=False), times)

    extraction_function = lambda x: torch.max(x, -1)[1].cpu()
    in_between(model, {"train":train_dataloader, "validation":validation_dataloader}, device, extraction_function)
main()
