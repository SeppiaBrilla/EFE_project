from typing import Callable
import numpy as np
import pandas as pd
from .base_predictor import Predictor, Predictor_initializer
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

class Metrics_initializer(Predictor_initializer):
    def __init__(self, order:'list[str]', idx2comb:'dict') -> None:
        super().__init__()
        self.order = order
        self.idx2comb = idx2comb
        self.comb2idx = {v:k for k,v in idx2comb.items()}

class Metrics_predictor(Predictor):

    def __init__(self, training_data:'list[dict]|None', 
                 idx2comb:'dict[int,str]|None',
                 features:'pd.DataFrame|None', 
                 metrics_type:'Literal["recall","precision","f1","accuracy"]|Callable' = "recall"
        ) -> 'None':
        """
        initialize an instance of the class Recall_predictor.
        ---------
        Parameters
            training_data:list[dict].
                Indicates the data to use to create the ordering used to break ties
            idx2comb:dict[int,str].
                A dictionary that, for each index, returns the corresponding combination
            features:pd.DataFrame.
                a dataframe with a column indicating the instances and a feature set for each feature
            metrics_type:Literal["recall","precision","f1","accuracy"]|Callable. Default = "recall"
                Indicates the metric to maximize in the ordering.
        -----
        Usage
        ```py
        train_data = [{"inst": "instance name", "trues":[1,0,1,0]}]
        fatures = pd.DataFrame([{"inst": "instance name", "feat1":0, "feat2":0, "feat3":1, "feat4":1}])
        idx2comb = {0: "combination_0", 1:"combination_1"}
        predictor = Recall_predictor(train_data, idx2comb, features)
        ```
        """
        super().__init__()
        if training_data is None or idx2comb is None or features is None:
            return

        self.idx2comb = idx2comb
        self.comb2idx = {v:k for k,v in idx2comb.items()}

        metric_value = {}
        metric = self.__get_metric(metrics_type)
        true_values = []
        predicted_values = []
        for datapoint in training_data:
            true_values.append(datapoint["trues"])
            predicted_values.append(np.round(features[features["inst"] == datapoint["inst"]].to_numpy()[0][1:].tolist()))
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)
        for i in idx2comb.keys():
            metric_value[idx2comb[i]] = metric(true_values[:,i], predicted_values[:,i])

        self.order = [k for k, _ in sorted(metric_value.items(), key =lambda x: x[1],  reverse=True)]
    

    @staticmethod
    def from_pretrained(pretrained:'Metrics_initializer') -> 'Metrics_predictor':
        predictor = Metrics_predictor(None, None, None)
        predictor.order = pretrained.order
        predictor.idx2comb = pretrained.idx2comb
        predictor.comb2idx = pretrained.comb2idx
        return predictor

    def __get_metric(self, metric) -> 'Callable':
        if metric == "recall":
            return recall_score
        elif metric == "accuracy":
            return accuracy_score
        elif metric == "f1":
            return f1_score
        elif metric == "precision":
            return precision_score
        elif isinstance(metric, Callable):
            return metric
        else:
            raise Exception(f"metric {metric} of type {type(metric)} can't be used as a valid metric")

    def __get_prediction(self, options:'list'):
            for candidate in self.order:
                if candidate in options:
                    return candidate

    def __get_dataset(self, dataset:'list') -> 'list[dict]':
        if type(dataset[0]) == float:
            return [{"inst":"", "features":dataset}]
        return dataset

    def predict(self, dataset:'list[dict]|list[float]', filter:'bool'=False) -> 'list[dict]|str':
        """
        Given a dataset, return a list containing each prediction for each datapoint and the sum of the total predicted time.
        -------
        Parameters
            dataset:list[dict]
                A list containing, for each datapoint to predict, a list of features to use for the prediction, a dictionary containing, for each option, the corresponding time
        ------
        Output
            A tuple containing:
                - a list of dicts with, for each datapoint, the chosen option and the corresponding predicted time
                - a float corresponding to the total time of the predicted options
        """

        is_single = type(dataset[0]) == float
        dataset = self.__get_dataset(dataset)

        if len(dataset[0]["features"]) != len(list(self.idx2comb.keys())):
            raise Exception(f"number of features is different from number of combinations: {len(dataset[0]['features'])} != {len(list(self.idx2comb.keys()))}")

        predictions = []
        for datapoint in dataset:
            options = list(self.idx2comb.values())
            if filter:
                options = [o for o in self.comb2idx.keys() if datapoint["features"][self.comb2idx[o]] < .5]
                if len(options) == 0:
                    options = list(self.idx2comb.values())
            chosen_option = self.__get_prediction(options)
            predictions.append({"chosen_option": chosen_option, "inst": datapoint["inst"]})

        if is_single:
            return predictions[0]["chosen_option"]

        return predictions

