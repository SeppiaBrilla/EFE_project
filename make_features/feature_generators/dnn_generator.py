from .base_generator import Generator
import torch.nn as nn
import torch.nn.functional as F
from torch import load, device, cuda
from transformers import AutoModel, AutoTokenizer, logging
logging.set_verbosity_error()

class Model(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained("tororoin/longformer-8bitadam-2048-main")
        self.output_layer = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        out = self.output_layer(encoded_input)
        out = F.sigmoid(out)
        return {"out": out.cpu().tolist()[0], "language_model":encoded_input.cpu().tolist()[0]}

class Language_features_generator(Generator):
    def __init__(self, names:'list', pre_trained_weights:'str', probabilities_only:'bool'=False) -> None:
        super().__init__()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.model = Model(len(names))
        self.model.load_state_dict(load(pre_trained_weights))
        self.model = self.model.to(self.device)
        self.names = names
        self.tokenizer = AutoTokenizer.from_pretrained("tororoin/longformer-8bitadam-2048-main")
        self.probabilities_only = probabilities_only

    def generate(self, instance: 'str') -> 'dict[str,float]':
        tokenized_instance = self.tokenizer(instance, truncation=True, return_tensors="pt")
        tokenized_instance = {k:tokenized_instance[k].to(self.device) for k in tokenized_instance.keys()}
        model_output = self.model(tokenized_instance)
        if self.probabilities_only:
            return {self.names[i]: model_output["out"][i] for i in range(len(self.names))}
        else:
            out = {self.names[i]: model_output["out"][i] for i in range(len(self.names))}
            for i in range(len(model_output["language_model"])):
                out[f"feat{i}"] = model_output["language_model"][i]
            return out

