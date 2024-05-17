from .base_predictor import Predictor, Predictor_initializer
from sys import stderr
import platform
import concurrent.futures
import re
import subprocess
import pandas as pd
import os

class Autofolio_initializer(Predictor_initializer):
    def __init__(self, model:'str', max_threads:'int') -> None:
        super().__init__()
        self.model = model
        self.max_threads = max_threads

class Autofolio_predictor(Predictor):
    
    MODEL_NAME = "autofolio_random_forest_model"
    CACHE_DIR = ".cache"

    def __init__(self, training_data:'list[dict]|None', 
                 features:'pd.DataFrame|None', 
                 max_threads:'int' = 12,
                 pre_trained_model:'str|None' = None
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
            hyperparameters:dict|None. Default=None
                hyperparameters of the clustering model to use. If None, a greedy search will be done to find the best hyperparameters
        -----
        Usage
        ```py
        train_data = [{"inst": "instance name", "trues":[1,0,1,0]}]
        fatures = pd.DataFrame([{"inst": "instance name", "feat1":0, "feat2":0, "feat3":1, "feat4":1}])
        idx2comb = {0: "combination_0", 1:"combination_1"}
        predictor = Clustering_predictor(train_data, idx2comb, features)
        ```
        """

        super().__init__()
        if training_data is None or features is None:
            return
        if "3.6" not in platform.python_version():
            raise Exception(f"AUtofolio works only on python version 3.6.x. found {platform.python_version()}")

        self.max_threads = max_threads
        if pre_trained_model is None:
            if not os.path.isdir(self.CACHE_DIR):
                os.makedirs(self.CACHE_DIR)

            times_file = os.path.join(self.CACHE_DIR, f"train_times.csv")
            pre_trained_model = os.path.join(self.CACHE_DIR, self.MODEL_NAME)
            features_file = os.path.join(self.CACHE_DIR, f"train_features.csv")

            x_header = list(features.columns)
            x_train_file = self.__create_file(features_file)
            x_train_file.write(",".join(x_header) + "\n")
            x_train = [[str(f) for f in features[features["inst"] == datapoint["inst"]].to_numpy()[0].tolist()] for datapoint in training_data]
            self.__save(x_train, x_train_file)
            x_train_file.close()
            combinations = sorted(list(training_data[0]["times"].keys()))
            y_header = ["inst"] + combinations
            y_train_file = self.__create_file(times_file)
            y_train_file.write(",".join(y_header) + "\n")
            y_train = [[datapoint["inst"]] + [str(datapoint["times"][comb]) for comb in combinations] for datapoint in training_data]
            self.__save(y_train, y_train_file)
            y_train_file.close()

            subprocess.run(
                ["python", "AutoFolio/scripts/autofolio", 
                 "--performance_csv", times_file, "--feature_csv", features_file, "--save", pre_trained_model])
        self.model = pre_trained_model

    @staticmethod
    def from_pretrained(pretrained:'Autofolio_initializer') -> 'Autofolio_predictor':
        predictor = Autofolio_predictor(None, None)
        predictor.model = pretrained.model
        predictor.max_threads = pretrained.max_threads
        return predictor

    def __create_file(self, file_name):
        f = open(file_name, "w")
        f.write("")
        f.close()
        return open(file_name, "a")

    def __save(self, data, file):
        for d in data:
            file.write(f"{','.join(d)}\n")

    def __get_prediction(self, options:'list', inst:'str'):
        options = [str(o) for o in options]
        out = subprocess.run(['python3', 'AutoFolio/scripts/autofolio', '--load', self.model, '--feature_vec', f'{" ".join(options)}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = out.stdout.decode('utf-8')
        chosen_option = re.findall(r"\[\('([a-zA-Z0-9.,-_]*)', [0-9]*\)\]", out)
        if len(chosen_option) == 0:
            raise Exception(out)
        return chosen_option[0], inst

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

        if filter != False:
            print("WARNING: predictor Autofolio cannot pre-filter option", file=stderr)
        predictions = []
        with concurrent.futures.ThreadPoolExecutor(self.max_threads) as executor:
            futures = {executor.submit(self.__get_prediction,datapoint["features"], datapoint["inst"]): datapoint["inst"] for datapoint in dataset}

            for future in concurrent.futures.as_completed(futures):
                text = futures[future]
                try:
                    result = future.result()
                    predictions.append({"chosen_option": result[0], "inst": result[1]})
                except Exception as e:
                    print(f"An error occurred for text '{text}': {e}", file=stderr)

        if is_single:
            return predictions[0]["chosen_option"]
        return predictions
