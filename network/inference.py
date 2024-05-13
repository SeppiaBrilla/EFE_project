import torch.functional as F
import torch
from json import loads
import torch.nn.functional as F
from helper import dict_lists_to_list_of_dicts
from models import BaseModel, get_tokenizer
from sys import argv
from tqdm import tqdm

class Feature_model(BaseModel):
    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        return torch.cat((encoded_input, F.sigmoid(self.output_layer(encoded_input))), dim=1)
        # return F.sigmoid(self.output_layer(encoded_input))

def main():

    if argv[1] == '--help':
        print(f"{argv[0]} dataset batch_size pretrained_weights")
        return

    dataset, pretrained_weights, save_file = argv[1], argv[2], argv[3]

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
    instances = [d["instance_value_json"] for d in data]
    y = [d["instance_name"] for d in data]
    for i in range(len(y)):
        assert data[i]["instance_name"] == y[i] and data[i]["instance_value_json"] == instances[i]
    x = dict_lists_to_list_of_dicts(tokenizer(instances, padding=True, truncation=True, return_tensors='pt'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    length = len(data[0]["all_times"])
    model = Feature_model(bert_type, length, dropout=.3)
    model.load_state_dict(torch.load(pretrained_weights))
    heading = ["inst"] + [f"feat_{i}" for i in range(model.bert.config.hidden_size)] + [f"prob_{i}" for i in range(length)]
    # heading = ["inst"] + [f"prob_{i}" for i in range(length)]
    heading = ",".join(heading)
    final_csv = f"{heading}\n"
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i in tqdm(range(len(x))):
            input = x[i]
            input = {key: input[key].to(device).reshape((1, input[key].size()[0])) for key in input.keys()}
            result = model(input)
            result = result.tolist()[0]
            assert len(result) == model.bert.config.hidden_size + length
            # assert len(result) == length
            final_csv += y[i] + "," + ",".join([str(r) for r in result]) + "\n"
    f = open(save_file, "w")
    f.write(final_csv)
    f.close()


main()
