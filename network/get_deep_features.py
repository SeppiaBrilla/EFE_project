import torch
import numpy as np
from json import loads
from helper import dict_lists_to_list_of_dicts, get_dataloader, remove_comments, to
from models import Timeout_and_selection_model, get_tokenizer
from sys import argv
class MY_MODEL(Timeout_and_selection_model):
    
    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        raw_timeouts = self.timeouts_layer(encoded_input)
        timeouts = self.sigmoid(raw_timeouts)

        selection = self.intermidiate(torch.concatenate((encoded_input, timeouts), 1))
        selection = self.relu(selection)
        selection = self.model_selection_layer(selection)
        selection = torch.nn.functional.softmax(selection, dim=1)
        out = torch.cat((encoded_input, selection, timeouts), 1)
        return out

def main():
    
    if argv[1] == '--help':
        print(f"{argv[0]} dataset, batch_size, bert_type, pretrained_weights, feature_file")
        print("bert types: \n[1]FacebookAI/roberta-base \n[2]bert-base-uncased \n[3]allenai/longformer-base-4096")
        return
    
    dataset, batch_size, bert_type, pretrained_weights, feature_file = argv[1], int(argv[2]), argv[3], argv[4], argv[5]
    
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
    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt', max_length = 2048))
    y = [d["instance_name"] for d in data]
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, batch_size, [9])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    length = 16
    model = MY_MODEL(bert_type, length)
    model.load_state_dict(torch.load(pretrained_weights))
    
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for idx, (xc, yc) in enumerate(train_dataloader):
            xc = to(xc, device)
            out = model(xc)
            
            out = out.cpu().tolist()
            for i in range(len(out)):
                print(f"{yc[i]},{','.join([str(o) for o in out[i]])}")
            del out
            del xc
main()
