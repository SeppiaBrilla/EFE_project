import torch.nn as nn
import torch
from json import loads, dumps
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from neuralNetwork import In_between_epochs
from models import BaseModel, get_tokenizer
from helper import  dict_lists_to_list_of_dicts, one_hot_encoding, remove_comments, get_dataloader, get_time_matrix
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

    ### import the data
    f = open(dataset)
    data = loads(f.read())
    f.close()
    ### data manipulation
    for datapoint in data:
        datapoint["all_times"] = [0 if d["time"] < 3600 else 1 for d in datapoint["all_times"]]

    all_times = [datapoint["all_times"] for datapoint in data]
    combinations = [d["combination"] for d in data[0]["all_times"]]
    
    ### dataset creation
    
    tokenizer = get_tokenizer(bert_type)
    instances = [remove_comments(datapoint['instance']) for datapoint in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances, padding=True, truncation=True, return_tensors='pt'))
    y = [datapoint["combination"] for datapoint in data]
    y = one_hot_encoding(y, combinations)
    
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, batch_size)
    
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
    ### training 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    length = len(combinations)
    model = BaseModel(bert_type, length, dropout=.3)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))

    extraction_function = lambda x: torch.max(x, -1)[1].cpu().tolist()
    train_score, val_score = model.train_network(train_dataloader, 
                    validation_dataloader, 
                    torch.optim.SGD, 
                    loss_function=nn.CrossEntropyLoss(),
                    device=device, 
                    verbose=True, 
                    output_extraction_function= extraction_function, 
                    metrics={"accuracy": accuracy_score, "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")},
                    learning_rate=learning_rate,
                    in_between_epochs={"validate_times":in_between},
                    epochs=epochs)

    torch.save(model.state_dict(), save_weights_file)

    for key in train_score:
                train_score[key] = [float(v) for v in train_score[key]]
                val_score[key] = [float(v) for v in val_score[key]]
    f = open(history_file, 'w')
    f.write(dumps({"train":train_score, "validation":val_score}))
    f.close()

    ### testing

    print("evaluating on the training, validation and test set")

    model = model.to(device)
    model.eval()
    train_prediction = model.predict(train_dataloader, extraction_function, device)
    validation_prediction = model.predict(validation_dataloader, extraction_function, device)
    test_prediction = model.predict(test_dataloader, extraction_function, device)

    len_train = len(train_dataloader.dataset)
    len_val = len(validation_dataloader.dataset)
    len_test = len(test_dataloader.dataset)

    times_matrix = get_time_matrix(np.array([y_tensor.tolist() for y_tensor in y]).shape, all_times)
    min_train = [min(times_matrix[i, :]) for i in range(len_train)]
    min_val = [min(times_matrix[i, :]) for i in range(len_train, len_train + len_val)]
    min_test = [min(times_matrix[i, :]) for i in range(len_train + len_val, len_train + len_val + len_test)]

    pred_train = [times_matrix[i, train_prediction[i]] for i in range(len_train)]
    pred_val = [times_matrix[len_train + i, validation_prediction[i]] for i in range(len_val)]
    pred_test = [times_matrix[len_train + len_val + i, test_prediction[i]] for i in range(len_test)]

    print(f"train set:\nvb:{sum(min_train)}     pred:{sum(pred_train)}")
    print(f"validation set:\nvb:{sum(min_val)}     pred:{sum(pred_val)}")
    print(f"test set:\nvb:{sum(min_test)}     pred:{sum(pred_test)}")
    
    f = open(prediction_file, "w")
    f.write(dumps({"train": train_prediction, "validation":validation_prediction, "test":test_prediction}))
    f.close()
main()
