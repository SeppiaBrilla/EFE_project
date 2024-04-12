import torch.nn as nn
import torch
from helper import get_dataloader, dict_lists_to_list_of_dicts, remove_comments, save_predictions
from models import get_tokenizer, BaseModel
from json import loads
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from sys import argv
from json import dump

def loss(input: torch.Tensor, target: torch.Tensor):
    input = F.sigmoid(input)
    epsilon = 1e-12  

    bce_loss = F.binary_cross_entropy(input, target, reduction='none')

    fn_mask = (input < target).float()
    weighted_fn_loss = (bce_loss + epsilon) * (1 + 0.1)

    fp_mask = (input > target).float()
    weighted_fp_loss = (bce_loss + epsilon) * (1 - 0.1)

    total_loss = (bce_loss * (1 - fn_mask) * (1 - fp_mask)) + (weighted_fn_loss * fn_mask) + (weighted_fp_loss * fp_mask)

    return torch.mean(total_loss)

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

    combinations = [d["combination"] for d in data[0]["all_times"]]

    tokenizer = get_tokenizer(bert_type)
    instances_and_model = [remove_comments(d["instance_value"]) for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = []
    for datapoint in data:
        y_point = [0 if d["time"] < 3600 else 1 for d in datapoint["all_times"]]
        y.append(torch.Tensor(y_point))

    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, 4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    length = len(combinations)
    model = BaseModel(bert_type, length, dropout=.3)
    sigmoid = nn.Sigmoid()
    extraction_function = lambda x: torch.round(sigmoid(x)).detach().cpu().view(-1)


    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))

    train_history_data, validation_history_data = model.train_network(train_dataloader, 
        validation_dataloader, 
        torch.optim.SGD, 
        loss_function=loss,
        device=device, 
        verbose=True, 
        output_extraction_function= extraction_function, 
        batch_size=batch_size,
        metrics={
        "accuracy": accuracy_score, 
        "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")},
        learning_rate=learning_rate,
        epochs=epochs)
    
    torch.save(model.state_dict(), save_weights_file)
    
    f = open(history_file, "w")
    dump({"train": train_history_data, "validation": validation_history_data}, f)
    f.close()
    save_predictions(model, {"train": train_dataloader, "validation":validation_dataloader, "test":test_dataloader}, prediction_file, device)
