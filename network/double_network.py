import torch.functional as F
import torch
from json import loads
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from neuralNetwork import In_between_epochs
import torch.nn.functional as F
from helper import dict_lists_to_list_of_dicts, get_dataloader, get_time_matrix, remove_comments, save_predictions
from models import Timeout_and_selection_model, get_tokenizer
from sys import argv
 
class Evaluate_time(In_between_epochs):

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
        self.prev_pred_train = np.Infinity
        self.prev_pred_val = np.Infinity
        self.prev_pred_test = np.Infinity
        
    def __call__(self, model, loaders, device, output_extraction_function):
        train_prediction = model.predict(self.train_dataloader, output_extraction_function, device)
        validation_prediction = model.predict(loaders["validation"], output_extraction_function, device)
        test_prediction = model.predict(self.test_dataloader, output_extraction_function, device)
        pred_train = round(sum([self.times_matrix[i, train_prediction[i]] for i in range(self.len_train)]), 2)
        pred_val = round(sum([self.times_matrix[self.len_train + i, validation_prediction[i]] for i in range(self.len_val)]), 2)
        pred_test = round(sum([self.times_matrix[self.len_train + self.len_val + i, test_prediction[i]] for i in range(self.len_test)]), 2)
        tr_r, v_r, te_r = self.get_arrows(pred_train, pred_val, pred_test)

        print("vb   pred    sb time  unique values")
        print(f"train set:\n{self.vb_train}     {pred_train}{tr_r}      {self.sb_train}    {np.unique(train_prediction)}")
        print(f"validation set:\n{self.vb_validation}     {pred_val}{v_r}     {self.sb_validation}    {np.unique(validation_prediction)}")
        print(f"test set:\n{self.vb_test}     {pred_test}{te_r}     {self.sb_test}  {np.unique(test_prediction)}")
        print(f"\n{'-'*100}\n")

        self.prev_pred_train = pred_train
        self.prev_pred_val = pred_val
        self.prev_pred_test = pred_test
        return False
    
    def get_arrows(self, pred_train, pred_val, pred_test):
        tr_r, v_r, te_r = "", "", ""
        if self.prev_pred_train < pred_train:
            tr_r = "/\\"
        elif self.prev_pred_train > pred_train:
            tr_r = "\\/"
        else:
            tr_r = "="
        
        if self.prev_pred_val < pred_val:
            v_r = "/\\"
        elif self.prev_pred_val > pred_val:
            v_r = "\\/"
        else:
            v_r = "="

        if self.prev_pred_test < pred_test:
            te_r = "/\\"
        elif self.prev_pred_test > pred_test:
            te_r = "\\/"
        else:
            te_r = "="

        return tr_r, v_r, te_r
def main():
    
    if argv[1] == '--help':
        print("network.py dataset batch_size bert_type epochs learning_rate history_file prediction_file save_weights_file pretrained_weights")
        print("bert types: \n[1]FacebookAI/roberta-base \n[2]bert-base-uncased \n[3]allenai/longformer-base-4096")
        return
    
    dataset, batch_size, bert_type, epochs, learning_rate, history_file, prediction_file, save_weights_file = argv[1], int(argv[2]), argv[3], int(argv[4]), float(argv[5]), argv[6], argv[7], argv[8]
    
    pretrained_weights = None
    if len(argv) == 10:
        pretrained_weights = argv[9]

    if bert_type == "1":
        bert_type = "FacebookAI/roberta-base"
    elif bert_type == "2":
        bert_type = "bert-base-uncased"
    elif bert_type == "3":
        bert_type = "allenai/longformer-base-4096"

    f = open(dataset)
    data = loads(f.read())
    f.close()

    tokenizer = get_tokenizer(bert_type)
    instances_and_model = [remove_comments(d["instance_value"]) for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = []

    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
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
        
    all_times = [datapoint["all_times"] for datapoint in data]
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, 4, [9])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    length = len(combinations)
    model = Timeout_and_selection_model(bert_type, length, dropout=.3)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))
 
    extraction_function = lambda x: torch.max(x["algorithm_selection"], -1)[1].cpu()

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
                test_dataloader, train_dataloader, times)

    def loss(y_pred, y_true):
        timeout_loss = F.binary_cross_entropy_with_logits(y_pred["timeouts"], y_true["timeouts"])
        algorithm_selection_loss = F.cross_entropy(y_pred["algorithm_selection"], F.softmax(y_true["algorithm_selection"], dim=1))
        max_values = [max([float(y_true["times"][j][i]) for j in range(length) if y_true["times"][j][i] < 3600]) for i in range(len(y_true["times"][0]))]
        min_values = [min([float(y_true["times"][j][i]) for j in range(length) ]) for i in range(len(y_true["times"][0]))]
        weights = [[1. for _ in range(length)] for _ in range(len(y_true["times"][0]))]
        # weights = 1 + torch.tensor([
        #         [(float(y_true["times"][j][i]) - min_values[j]) / (max_values[j] - min_values[j]) 
        #             if y_true["times"][j][i] < 3600 else 2 
        #             for j in range(length)] 
        #         for i in range(len(y_true["times"][0]))
        #     ])
        for j in range(len(y_true["times"][0])):
            for i in range(length):
                weights[j][i] += (float(y_true["times"][i][j]) - min_values[j]) / (max_values[j] - min_values[j])

        weights = torch.tensor(weights)
        weights = weights.to(device)
        combined = -torch.sum(y_true["algorithm_selection"] * torch.log_softmax(y_pred["algorithm_selection"], dim=1) * weights, dim=1) 
        combined = torch.mean(combined)
        del weights
        return combined

    train_data, validation_data =   model.train_network(train_dataloader, 
                    validation_dataloader, 
                    torch.optim.SGD, 
                    loss_function=loss,
                    device=device, 
                    batch_size=batch_size,
                    verbose=True, 
                    output_extraction_function= extraction_function, 
                    metrics={
                     "accuracy": accuracy_score, 
                     "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")},
                    in_between_epochs={"validate_times":in_between},
                    learning_rate=learning_rate,
                    epochs=epochs)

    torch.save(model.state_dict(), save_weights_file)
    from json import dump
    f = open(history_file, 'w')
    for key in train_data:
            train_data[key] = [float(v) for v in train_data[key]]
            validation_data[key] = [float(v) for v in validation_data[key]]
    dump({"train": train_data, "validation": validation_data}, f)
    f.close()
    save_predictions(model, {"train": train_dataloader, "validation":validation_dataloader, "test":test_dataloader}, prediction_file, device,
                     extraction_function= lambda x: x["algorithm_selection"].tolist())
main()
