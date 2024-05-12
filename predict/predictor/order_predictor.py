from .base_predictor import Predictor
from typing import Callable

class Static_ordering_predictor(Predictor):

    def __init__(self, training_data:'list[dict]', 
                 idx2comb:'dict[int,str]', 
                 ordering_type:'Literal["single_best","wins"]|Callable' = "single_best"
        ) -> 'None':
        """
        initialize an instance of the class Recall_predictor.
        ---------
        Parameters
            training_data:list[dict].
                Indicates the data to use to create the ordering used to break ties
            idx2comb:dict[int,str].
                A dictionary that, for each index, returns the corresponding combination
            ordering_type:Literal["single_best","wins"]!Callable. Default="single_best"
                The strategy to use to create the static scheduling:
                    - single_best: order the options using their total runtime
                    - wins: order the options using the amount of times a given option was the best
                    - Callable: a function that, given the training data, returns a list to use as ordering
        -----
        Usage
        ```py
        train_data = [{"inst": "instance name", "times":{"option1":1, "option2":2}, "time":1}]
        idx2comb = {0: "combination_0", 1:"combination_1"}
        predictor = Static_ordering_predictor(train_data, idx2comb)
        ```
        """
        super().__init__()

        self.idx2comb = idx2comb
        self.comb2idx = {v:k for k,v in idx2comb.items()}

        if ordering_type == "single_best":
            self.order = self.__get_single_best_ordering(training_data, idx2comb)
        elif ordering_type == "wins":
            self.order = self.__get_wins_ordering(training_data, idx2comb)
        elif isinstance(ordering_type,Callable):
            self.order = ordering_type(training_data)
        else:
            raise Exception(f"ordering_type {ordering_type} of type {type(ordering_type)} not supported")

    def __get_single_best_ordering(self, training_data:'list[dict]', idx2comb:'dict') -> 'list[str]':
        order = {comb:0 for comb in idx2comb.values()}
        for datapoint in training_data:
            for combination in datapoint["times"].keys():
                order[combination] += datapoint["times"][combination]
        
        return [combination for combination, _ in sorted(order.items(), key= lambda x: x[1])]

    def __get_wins_ordering(self, training_data:'list[dict]', idx2comb:'dict') -> 'list[str]':
        order = {comb:0 for comb in idx2comb.values()}
        for datapoint in training_data:
            combination, _ = min(datapoint["times"].items(), key=lambda x: x[1])
            order[combination] += 1
        
        return [combination for combination, _ in sorted(order.items(), key= lambda x: x[1], reverse=True)]

    def __get_prediction(self, options:'list'):
        for candidate in self.order:
            if candidate in options:
                return candidate

    def predict(self, dataset:'list[dict]', filter:'bool' = False) -> 'tuple[list[dict[str,str|float]],float]':
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
        if len(dataset[0]["features"]) != len(list(self.idx2comb.keys())):
            raise Exception(f"number of features is different from number of combinations: {len(dataset[0]['features'])} != {len(list(self.idx2comb.keys()))}")

        predictions = []
        total_time = 0
        for datapoint in dataset:
            options = list(self.idx2comb.values())
            if filter:
                options = [o for o in self.comb2idx.keys() if datapoint["features"][self.comb2idx[o]] < .5]
                if len(options) == 0:
                    options = list(self.idx2comb.values())
            chosen_option = self.__get_prediction(options)
            time = datapoint["times"][chosen_option]
            total_time += time
            predictions.append({"choosen_option": chosen_option, "time": time})

        return predictions, total_time
