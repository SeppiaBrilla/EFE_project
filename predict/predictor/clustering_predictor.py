from .base_predictor import Predictor, Predictor_initializer, isnan
from tqdm import tqdm
from sys import stderr
import concurrent.futures
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
import joblib
from time import time

class kmeans_initializer(Predictor_initializer):
    def __init__(self, pretrained_clustering_file_path:'str', order:'dict', idx2comb:'dict') -> None:
        super().__init__()
        self.kmeans = joblib.load(pretrained_clustering_file_path)
        self.order = order
        self.idx2comb = idx2comb

class Kmeans_predictor(Predictor):

    MODEL_NAME = "kmeans.pkl"

    def __init__(self, training_data:'list[dict]|None', 
                 idx2comb:'dict[int,str]|None', 
                 features:'pd.DataFrame|None', 
                 hyperparameters:'dict|None' = None,
                 max_threads:'int' = 12
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
        if training_data is None or idx2comb is None or features is None:
            return
        self.idx2comb = idx2comb
        self.comb2idx = {v:k for k,v in idx2comb.items()}
        TRAIN_ELEMENTS = int(len(training_data) * .9)
        train = training_data[:TRAIN_ELEMENTS]
        validation = training_data[TRAIN_ELEMENTS:]
        self.max_threads = max_threads

        training_features = [[str(f) for f in features[features["inst"] == datapoint["inst"]].to_numpy()[0].tolist()] for datapoint in training_data]
        times = {opt:0 for opt in training_data[0]["times"].keys()}
        for i in range(len(training_data)):
            inst = training_data[i]["inst"]
            training_features[i].pop(training_features[i].index(inst))
            for key in training_data[i]["times"].keys():
                times[key] += training_data[i]["times"][key]

        self.sb = min(times.items())[0]

        training_features = np.array(training_features)


        if hyperparameters is None:
            hyperparameters = self.__get_clustering_parameters(training_features, train, validation, features, idx2comb)
        self.clustering_parameters = hyperparameters
        self.clustering_model = KMeans(**hyperparameters)
        y_pred = self.clustering_model.fit_predict(training_features)
        cluster_range = range(hyperparameters["n_clusters"])
        stats = {i: {comb:0 for comb in idx2comb.values()} for i in cluster_range}
        for i in range(len(train)):
            for option in train[i]["times"].keys():
                stats[y_pred[i]][option] += train[i]["times"][option]
        self.order = {str(i): {k:v for k, v in sorted(stats[i].items(), key=lambda item: item[1], reverse=False)} for i in cluster_range}

    def __get_clustering_parameters(self, 
                                    training_features:'np.ndarray', 
                                    train_data:'list[dict]', 
                                    validation_data:'list[dict]', 
                                    features:'pd.DataFrame',
                                    idx2comb:'dict') -> 'dict':
        parameters = list(ParameterGrid({
            'n_clusters': range(2, 21),
            'init': ['k-means++', 'random'],
            'max_iter': [100, 200, 300],
            'tol': [1e-3, 1e-4, 1e-5],
            'n_init': [5, 10, 15, "auto"],
            'random_state': [42],
            'verbose': [0]
        }))
        clusters_val = []
        
        with concurrent.futures.ThreadPoolExecutor(self.max_threads) as executor:
            futures = {executor.submit(
                self.__get_clustering_score, 
                params, training_features, 
                idx2comb, train_data, 
                validation_data, features): json.dumps(params)
            for params in parameters}

            for future in tqdm(concurrent.futures.as_completed(futures)):
                params = futures[future]
                try:
                    result = future.result()
                    clusters_val.append((json.loads(params), result))
                except Exception as e:
                    print(f"An error occurred for text '{params}': {e}", file=stderr)

        best_cluster_val = min(clusters_val, key=lambda x: x[1])
        return best_cluster_val[0]
    
    def __get_clustering_score(self, params:'dict', 
                               training_features:'np.ndarray', 
                               idx2comb:'dict', 
                               train_data:'list[dict]', 
                               validation_data:'list', 
                               features:'pd.DataFrame'):
        kmeans = KMeans(**params)
        y_pred = kmeans.fit_predict(training_features)
        stats = {i: {comb:0 for comb in idx2comb.values()} for i in range(params["n_clusters"])}
        for i in range(len(train_data)):
            for option in train_data[i]["times"].keys():
                stats[y_pred[i]][option] += train_data[i]["times"][option]
        order = {str(i): {k:v for k, v in sorted(stats[i].items(), key=lambda item: item[1], reverse=False)} for i in range(params["n_clusters"])}
        time = 0
        for datapoint in validation_data:
            datapoint_features = features[features["inst"] == datapoint["inst"]].to_numpy()[0].tolist()
            datapoint_features.pop(datapoint_features.index(datapoint["inst"]))
            datapoint_features = np.array(datapoint_features)
            preds = kmeans.predict(datapoint_features.reshape(1, -1))
            datapoint_candidates = list(idx2comb.values())
            option = self.__get_prediction(datapoint_candidates, int(preds[0]), order)
            time += datapoint["times"][option]
        return time

    @staticmethod 
    def from_pretrained(pretrained:'kmeans_initializer') -> 'Kmeans_predictor':
        predictor = Kmeans_predictor(None, None, None, None)
        predictor.idx2comb = pretrained.idx2comb
        predictor.order = pretrained.order
        predictor.clustering_model = pretrained.kmeans
        predictor.clustering_parameters = {
            'n_clusters': pretrained.kmeans.n_clusters,
            'init': pretrained.kmeans.init,
            'max_iter': pretrained.kmeans.max_iter,
            'tol': pretrained.kmeans.tol,
            'n_init': pretrained.kmeans.n_init,
            'random_state': pretrained.kmeans.random_state,
            'verbose': [0]
        }
        return predictor

    def __get_dataset(self, dataset:'list') -> 'list[dict]':
        if type(dataset[0]) == float:
            return [{"inst":"", "features":dataset}]
        return dataset

    def __get_prediction(self, options:'list', category:'int', order:'dict|None' = None):
        order = order if not order is None else self.order
        for candidate in order[str(category)]:
            if candidate in options:
                return candidate

    def predict(self, dataset:'list[dict]|list[float]', filter:'bool'=False) -> 'list[dict]|dict':
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
        if filter and len(dataset[0]["features"]) != len(list(self.idx2comb.keys())):
            raise Exception(f"number of features is different from number of combinations: {len(dataset[0]['features'])} != {len(list(self.idx2comb.keys()))}")

        predictions = []
        for datapoint in tqdm(dataset):
            feat = np.array(datapoint["features"]).reshape(1,-1)
            start = time()
            if isnan(feat.tolist()[0]):
                chosen_option = self.sb
            else:
                category = self.clustering_model.predict(feat)
                options = list(self.idx2comb.values())
                if filter:
                    options = [o for o in self.comb2idx.keys() if datapoint["features"][self.comb2idx[o]] < .5]
                    if len(options) == 0:
                        options = list(self.idx2comb.values())
                chosen_option = self.__get_prediction(options, int(category[0]))
            predictions.append({"chosen_option": chosen_option, "inst": datapoint["inst"], "time": time() - start})

        if is_single:
            return predictions[0]
        return predictions
