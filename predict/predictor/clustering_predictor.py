from .base_predictor import Predictor, Predictor_initializer
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
import joblib

class Clustering_initializer(Predictor_initializer):
    def __init__(self, pretrained_clustering_file_path:'str', order:'dict', idx2comb:'dict') -> None:
        super().__init__()
        self.kmeans = joblib.load(pretrained_clustering_file_path)
        self.order = order
        self.idx2comb = idx2comb

class Clustering_predictor(Predictor):

    MODEL_NAME = "kmeans.pkl"

    def __init__(self, training_data:'list[dict]|None', 
                 idx2comb:'dict[int,str]|None', 
                 features:'pd.DataFrame|None', 
                 hyperparameters:'dict|None' = None,
                 filter:'bool|None' = True
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

        training_features = np.array([features[features["inst"] == datapoint["inst"]].to_numpy()[0][1:].tolist() for datapoint in train])
        if hyperparameters is None:
            hyperparameters = self.__get_clustering_parameters(training_features, train, validation, features, idx2comb, filter)
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
                                    idx2comb:'dict',
                                    filter:'bool') -> 'dict':
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
        for params in parameters:
            kmeans = KMeans(**params)
            y_pred = kmeans.fit_predict(training_features)
            stats = {i: {comb:0 for comb in idx2comb.values()} for i in range(params["n_clusters"])}
            for i in range(len(train_data)):
                for option in train_data[i]["times"].keys():
                    stats[y_pred[i]][option] += train_data[i]["times"][option]
            order = {i: {k:v for k, v in sorted(stats[i].items(), key=lambda item: item[1], reverse=False)} for i in range(params["n_clusters"])}
            time = 0
            for datapoint in validation_data:
                datapoint_features = features[features["inst"] == datapoint["inst"]].to_numpy()[0][1:]
                preds = kmeans.predict(datapoint_features.reshape(1, -1))
                datapoint_candidates = list(idx2comb.values())
                if filter:
                    datapoint_features = np.round(datapoint_features.tolist())
                    datapoint_candidates = [idx2comb[idx] for idx in idx2comb.keys() if datapoint_features[idx] == 1]
                option = self.__get_prediction(datapoint_candidates, int(preds[0]), order)
                time += datapoint["times"][option]
            clusters_val.append((params, time))

        best_cluster_val = min(clusters_val, key=lambda x: x[1])
        return best_cluster_val[0]

    @staticmethod 
    def from_pretrained(pretrained:'Clustering_initializer') -> 'Clustering_predictor':
        predictor = Clustering_predictor(None, None, None, None)
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

        if filter and len(dataset[0]["features"]) != len(list(self.idx2comb.keys())):
            raise Exception(f"number of features is different from number of combinations: {len(dataset[0]['features'])} != {len(list(self.idx2comb.keys()))}")

        predictions = []
        for datapoint in dataset:
            category = self.clustering_model.predict(np.array(datapoint["features"]).reshape(1,-1))
            options = list(self.idx2comb.values())
            if filter:
                options = [o for o in self.comb2idx.keys() if datapoint["features"][self.comb2idx[o]] < .5]
                if len(options) == 0:
                    options = list(self.idx2comb.values())
            chosen_option = self.__get_prediction(options, int(category[0]))
            predictions.append({"chosen_option": chosen_option, "inst": datapoint["inst"]})

        if is_single:
            return predictions[0]["chosen_option"]
        return predictions
