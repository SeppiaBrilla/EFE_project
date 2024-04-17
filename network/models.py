from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, LongformerModel, LongformerTokenizer, AutoTokenizer, AutoModel
import torch.nn as nn
import torch
from neuralNetwork import NeuralNetwork

class Timeout_and_selection_model(NeuralNetwork):
    def __init__(self, base_model_name, num_classes, dropout=.1) -> None:
        super().__init__()
        if "FacebookAI/roberta-base" == base_model_name:
            self.bert = RobertaModel.from_pretrained(base_model_name)
        elif "bert-base-uncased" == base_model_name:
            self.bert = BertModel.from_pretrained(base_model_name)
        elif "allenai/longformer-base-4096" == base_model_name:
            self.bert = LongformerModel.from_pretrained(pretrained_model_name_or_path = base_model_name)
        elif "microsoft/codebert-base" == base_model_name:
            self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        else:
            self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout)


        self.timeouts_layer = nn.Linear(self.bert.config.hidden_size, num_classes)

        self.intermidiate = nn.Linear(self.bert.config.hidden_size + num_classes, num_classes)

        self.model_selection_layer = nn.Linear(num_classes, num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        raw_timeouts = self.timeouts_layer(encoded_input)
        timeouts = self.sigmoid(raw_timeouts)

        timeouts = 1 - torch.round(timeouts)
        selection = self.intermidiate(torch.concatenate((encoded_input, timeouts), 1)) * timeouts
        selection = self.relu(selection)
        return {
            "algorithm_selection": self.model_selection_layer(selection),
            "timeouts": raw_timeouts
        }


class BaseModel(NeuralNetwork):
    def __init__(self, base_model_name, num_classes, dropout=.1) -> None:
        super().__init__()
        if "FacebookAI/roberta-base" == base_model_name:
            self.bert = RobertaModel.from_pretrained(base_model_name)
        elif "bert-base-uncased" == base_model_name:
            self.bert = BertModel.from_pretrained(base_model_name)
        elif "allenai/longformer-base-4096" == base_model_name:
            self.bert = LongformerModel.from_pretrained(pretrained_model_name_or_path = base_model_name)
        elif "microsoft/codebert-base" == base_model_name:
            self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        else:
            self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(self.bert.config.hidden_size, num_classes)


    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        return self.output_layer(encoded_input)

def get_tokenizer(bert_type:str):
    if "FacebookAI/roberta-base" == bert_type:
        return RobertaTokenizer.from_pretrained(bert_type)
    elif "bert-base-uncased" == bert_type:
        return BertTokenizer.from_pretrained(bert_type)
    elif "allenai/longformer-base-4096" == bert_type:
        return LongformerTokenizer.from_pretrained(bert_type)
    else:
        return AutoTokenizer.from_pretrained(bert_type)

