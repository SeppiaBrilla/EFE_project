import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from random import randint
from json import loads
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuralNetwork import In_between_epochs
import torch.nn.functional as F
from helper import dict_lists_to_list_of_dicts, get_dataloader, get_time_matrix, remove_comments
from models import BaseModel, get_tokenizer
from sys import argv

class Timeout_analiser(In_between_epochs):
    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, order, 
                 idx2comb, min_train, sb_train, min_val, sb_val, min_test, sb_test) -> None:
        super().__init__()
        self.train = train_dataloader
        self.val = validation_dataloader
        self.test = test_dataloader
        self.order = order
        self.idx2comb = idx2comb
        self.min_train = min_train
        self.sb_train = sb_train
        self.min_val = min_val
        self.sb_val = sb_val
        self.min_test = min_test
        self.sb_test = sb_test

    def custom_sort_func(self, x):
        for idx, (v, _) in enumerate(self.order):
            if v == x[0]:
                return idx
        return 10000000

    def analyse_prediction(self, preds, trues, idx2comb, order):
        def custom_sort_func(x):
            for idx, (v, _) in enumerate(order):
                if v == x[0]:
                    return idx
            return 10000000
        fp, fn, tp, tn = 0, 0, 0, 0
        undetected_timeouts, true_timeouts = 0, 0
        jt = 0
        good_oracle, bad_oracle, random_oracle, order_oracle = 0, 0, 0, 0
        for i in range(len(preds)):
            _, y_true = trues.dataset[i]
            y_pred = preds[i]
            if len(y_pred) == sum(y_pred):
                jt +=1
            remaining_times = [y_true["times"][idx2comb[j]] for j in range(len(y_pred)) if y_pred[j] == 0]
            true_timeouts += len([y_true["times"][key] for key in y_true["times"].keys() if y_true["times"][key] >= 3600])

            if len(remaining_times) == 0:
                remaining_times = [y_true["times"][idx2comb[j]] for j in range(len(y_pred))]
            undetected_timeouts += len([t for t in remaining_times if t >= 3600])
            good_oracle += min(remaining_times)
            bad_oracle += max(remaining_times)
            random_oracle += remaining_times[randint(0, len(remaining_times) - 1)]
            remaining_times_idxs = [(j, y_true["times"][idx2comb[j]]) for j in range(len(y_pred)) if y_pred[j] == 0]
            if len(remaining_times_idxs) == 0:
                remaining_times_idxs = [(j, y_true["times"][idx2comb[j]]) for j in range(len(y_pred))]
            remaining_times_idxs = sorted(remaining_times_idxs, key=custom_sort_func)
            order_oracle += remaining_times_idxs[0][1]
            for idx in range(len(y_pred)):
                if y_pred[idx] == y_true["competitivness"][idx]:
                    if y_pred[idx] == 1:
                        tp +=1
                    else:
                        tn +=1
                else:
                    if y_pred[idx] == 1:
                        fp +=1
                    else:
                        fn +=1
        return {"fp":fp, "fn":fn, "tp":tp, "tn":tn, 
                "jt":jt, "undetected_timeouts": undetected_timeouts, "true_timeouts": true_timeouts, 
                "good_oracle": good_oracle, "bad_oracle":bad_oracle, "random_oracle": random_oracle, "order_oracle": order_oracle
                }

    def __call__(self, model, loaders, device, output_extraction_function) -> bool:
        train_prediction = model.predict(self.train, output_extraction_function, device)
        validation_prediction = model.predict(self.val, output_extraction_function, device)
        test_prediction = model.predict(self.test, output_extraction_function, device)

        res = self.analyse_prediction(train_prediction, self.train, self.idx2comb, self.order)
        precision = res['tp'] / (res['tp'] + res['fp'])
        recall = res['tp'] / (res['tp'] + res['fn'])
        f1 = 2 * (precision * recall) / (precision + recall)
        total_timeouts = sum([sum(d[1]["competitivness"]) for d in self.train.dataset])

        print(f"""train set: 
        false positive: {res['fp']} false negative: {res['fn']} true positive: {res['tp']} true negative: {res['tn']}. 
        Just timeouts: {res['jt']} total element to discard: {total_timeouts} undetected timeouts: {res['undetected_timeouts']} true timeouts: {res['true_timeouts']}
        precision: {round(precision,2)} recall: {round(recall,2)} f1: {round(f1,2)}
        oracles:
        good oracle: {res['good_oracle']:,.2f} bad oracle: {res['bad_oracle']:,.2f} random oracle: {res['random_oracle']:,.2f} order: {res['order_oracle']:,.2f}
        virtual best: {self.min_train:,.2f} single best: {self.sb_train:,.2f} 
        good oracle/vb: {round(res['good_oracle']/self.min_train,2)} good oracle/sb: {round(res['good_oracle']/self.sb_train,2)} 
        bad oracle/vb: {round(res['bad_oracle']/self.min_train,2)} bad oracle/sb: {round(res['bad_oracle']/self.sb_train,2)} 
        random oracle/vb: {round(res['random_oracle']/self.min_train,2)} random oracle/sb: {round(res['random_oracle']/self.sb_train,2)} 
        order/vb: {round(res['order_oracle']/self.min_train,2)} order/sb: {round(res['order_oracle']/self.sb_train,2)} 
        """)

        res = self.analyse_prediction(validation_prediction, self.val, self.idx2comb, self.order)
        precision = res['tp'] / (res['tp'] + res['fp'])
        recall = res['tp'] / (res['tp'] + res['fn'])
        f1 = 2 * (precision * recall) / (precision + recall)
        total_timeouts = sum([sum(d[1]["competitivness"]) for d in self.val.dataset])
        print(f"""validation set: 
        false positive: {res['fp']} false negative: {res['fn']} true positive: {res['tp']} true negative: {res['tn']}. 
        Just timeouts: {res['jt']} total element to discard: {total_timeouts} undetected timeouts: {res['undetected_timeouts']} true timeouts: {res['true_timeouts']}
        precision: {round(precision,2)} recall: {round(recall,2)} f1: {round(f1,2)}
        oracles:
        good oracle: {res['good_oracle']:,.2f} bad oracle: {res['bad_oracle']:,.2f} random oracle: {res['random_oracle']:,.2f} order: {res['order_oracle']:,.2f}
        virtual best: {self.min_val:,.2f} single best: {self.sb_val:,.2f} 
        good oracle/vb: {round(res['good_oracle']/self.min_val,2)} good oracle/sb: {round(res['good_oracle']/self.sb_val,2)} 
        bad oracle/vb: {round(res['bad_oracle']/self.min_val,2)} bad oracle/sb: {round(res['bad_oracle']/self.sb_val,2)} 
        random oracle/vb: {round(res['random_oracle']/self.min_val,2)} random oracle/sb: {round(res['random_oracle']/self.sb_val,2)} 
        order/vb: {round(res['order_oracle']/self.min_val,2)} order/sb: {round(res['order_oracle']/self.sb_val,2)} 
        """)

        res = self.analyse_prediction(test_prediction, self.test, self.idx2comb, self.order)
        precision = res['tp'] / (res['tp'] + res['fp'])
        recall = res['tp'] / (res['tp'] + res['fn'])
        f1 = 2 * (precision * recall) / (precision + recall)
        total_timeouts = sum([sum(d[1]["competitivness"]) for d in self.test.dataset])
        print(f"""test set: 
        false positive: {res['fp']} false negative: {res['fn']} true positive: {res['tp']} true negative: {res['tn']}. 
        Just timeouts: {res['jt']} total element to discard: {total_timeouts} undetected timeouts: {res['undetected_timeouts']} true timeouts: {res['true_timeouts']}
        precision: {round(precision,2)} recall: {round(recall,2)} f1: {round(f1,2)}
        oracles:
        good oracle: {res['good_oracle']:,.2f} bad oracle: {res['bad_oracle']:,.2f} random oracle: {res['random_oracle']:,.2f} order: {res['order_oracle']:,.2f}
        virtual best: {self.min_test:,.2f} single best: {self.sb_test:,.2f} 
        good oracle/vb: {round(res['good_oracle']/self.min_test,2)} good oracle/sb: {round(res['good_oracle']/self.sb_test,2)} 
        bad oracle/vb: {round(res['bad_oracle']/self.min_train,2)} bad oracle/sb: {round(res['bad_oracle']/self.sb_test,2)} 
        random oracle/vb: {round(res['random_oracle']/self.min_test,2)} random oracle/sb: {round(res['random_oracle']/self.sb_test,2)} 
        order/vb: {round(res['order_oracle']/self.min_test,2)} order/sb: {round(res['order_oracle']/self.sb_test,2)} 
        """)

        return False

class Save_weights(In_between_epochs):
    def __init__(self, weights_name) -> None:
        super().__init__()
        self.name = weights_name

    def __call__(self, model, loaders, device, output_extraction_function) -> bool:
        torch.save(model.state_dict(), self.name)
        return False
def is_competitive(vb, option):
    return (option < 10 or vb * 2 <= option) and option < 3600

def main():

    if argv[1] == '--help':
        print(f"{argv[0]} dataset batch_size epochs learning_rate history_file save_weights_file fold pretrained_weights")
        #print("bert types: \n[1]FacebookAI/roberta-base \n[2]bert-base-uncased \n[3]allenai/longformer-base-4096 \n[4]microsoft/codebert-base")
        return

    dataset, batch_size, epochs, learning_rate, history_file, save_weights_file, fold = argv[1], int(argv[2]), int(argv[3]), float(argv[4]), argv[5], argv[6], int(argv[7])

    pretrained_weights = None
    if len(argv) == 9:
        pretrained_weights = argv[8]

    bert_type = "tororoin/longformer-8bitadam-2048-main"
    if bert_type == "1":
        bert_type = "FacebookAI/roberta-base"
    elif bert_type == "2":
        bert_type = "bert-base-uncased"
    elif bert_type == "3":
        bert_type = "allenai/longformer-base-4096"
    elif bert_type == "4":
        bert_type = "microsoft/codebert-base"
    f = open(dataset)
    data = loads(f.read())
    f.close()

    tokenizer = get_tokenizer(bert_type)
    instances_and_model = [d["instance_value_json"] for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = []

    idx2comb = {idx:comb["combination"] for idx, comb in enumerate(sorted(data[0]["all_times"], key= lambda x: x["combination"]))}
    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
    for datapoint in data:
        y_datapoint = sorted(datapoint["all_times"], key= lambda x: x["combination"])
        datapoint["all_times"] = y_datapoint
        ordered_times = [d["time"] for d in datapoint["all_times"]]
        ordered_times = sorted(ordered_times)
        vb = min([d["time"] for d in y_datapoint])
        competitivness = [0 if is_competitive(vb, d["time"]) else 1 for d in y_datapoint]
        y.append({
            "competitivness":torch.Tensor(competitivness),
            "times": {d["combination"]:d["time"] for d in y_datapoint}
        })
        
    all_times = [datapoint["all_times"] for datapoint in data]
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, 1, [fold])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    length = len(combinations)
    model = BaseModel(bert_type, length, dropout=.3)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))

    times = get_time_matrix((len(all_times), len(combinations)), all_times)

    len_train = len(train_dataloader.dataset)
    len_val = len(validation_dataloader.dataset)
    len_test = len(test_dataloader.dataset)
    sb_train = min([np.sum(times[:len_train, i]) for i in range(len(combinations))])
    order = [(i, np.sum(times[:len_train, i])) for i in range(len(combinations))]
    order = sorted(order, key=lambda x: x[1])
    sb_val = min([np.sum(times[len_train:len_train + len_val,i]) for i in range(len(combinations))])
    sb_test = min([np.sum(times[len_train + len_val:,i]) for i in range(len(combinations))])

    min_train = np.sum([min(times[i, :]) for i in range(len_train)])
    min_val = np.sum([min(times[len_train + i, :]) for i in range(len_val)])
    min_test = np.sum([min(times[len_train + len_val + i, :]) for i in range(len_test)])

    timeout_analiser = Timeout_analiser(DataLoader(train_dataloader.dataset, shuffle=False, batch_size=12),
                                        DataLoader(validation_dataloader.dataset, shuffle=False, batch_size=12), 
                                        DataLoader(test_dataloader.dataset, shuffle=False, batch_size=12), order, idx2comb,
                round(min_train, 2), round(sb_train, 2), round(min_val, 2), round(sb_val, 2), round(min_test, 2), 
                round(sb_test, 2))
    saver = Save_weights(save_weights_file)
    timeouts = [sum([1 if times[i, j] >= 3600 else 0 for i in range(len_train)]) for j in range(16)]
    max_timeouts = max(timeouts)
    timeouts = [10 * (1 - (timeout / max_timeouts)) for timeout in timeouts]
    weights = torch.tensor(timeouts)
    weights = weights.to(device)
    def loss(y_pred, y_true):
        timeouts_out = F.sigmoid(y_pred)
        timeouts = -(2 * y_true["competitivness"] * torch.log(timeouts_out) + (1 - y_true["competitivness"]) * torch.log(1 - timeouts_out))
        timeouts = timeouts * weights
        timeouts = torch.mean(timeouts)
        return timeouts

    def extraction_function(x):
        if isinstance(x, dict):
            x = x["competitivness"]
        return torch.round(torch.nn.functional.sigmoid(x)).cpu().tolist()

    train_data, validation_data =   model.train_network(train_dataloader, 
                    validation_dataloader, 
                    torch.optim.SGD, 
                    loss_function=loss,
                    device=device, 
                    batch_size=batch_size,
                    verbose=True, 
                    output_extraction_function= extraction_function, 
                    metrics={
                     "accuracy": lambda y_true, y_pred: accuracy_score(np.ravel(y_true), np.ravel(y_pred)), 
                     "f1_score": lambda y_true, y_pred: f1_score(np.ravel(y_true), np.ravel(y_pred), average="macro", zero_division=0),
                     "precision": lambda y_true, y_pred: precision_score(np.ravel(y_true), np.ravel(y_pred), average="macro", zero_division=0),
                     "recall": lambda y_true, y_pred: recall_score(np.ravel(y_true), np.ravel(y_pred), average="macro", zero_division=0)},
                    in_between_epochs={"validate_timeout":timeout_analiser, "save": saver},
                    learning_rate=learning_rate,
                    scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=10),
                    epochs=epochs)

    torch.save(model.state_dict(), save_weights_file)
    from json import dump
    f = open(history_file, 'w')
    for key in train_data:
            train_data[key] = [float(v) for v in train_data[key]]
            validation_data[key] = [float(v) for v in validation_data[key]]
    dump({"train": train_data, "validation": validation_data}, f)
    f.close()
main()
