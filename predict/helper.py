import argparse
from predictor.predictor import Predictor
from predictor.clustering_predictor import Clustering_predictor
from predictor.order_predictor import Static_ordering_predictor
from predictor.autofolio_predictor import Autofolio_predictor
from predictor.order_metrics import Metrics_predictor

def get_predictor(predictor_type:'str', 
                  train_data:'list[dict]', 
                  **kwargs) -> 'Predictor':
    if predictor_type == "static":
        if "ordering_type" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs an ordering type. ordering_type cannot be None")
        if "idx2comb" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs idx2comb. idx2comb cannot be None")
        return Static_ordering_predictor(idx2comb=kwargs["idx2comb"], training_data=train_data, ordering_type=kwargs["ordering_type"])
    elif predictor_type == "kmeans":
        if "idx2comb" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs idx2comb. idx2comb cannot be None")
        if "features" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs features. features cannot be None")
        hyperparameters =  kwargs["hperparameters"] if "hyperparameters" in kwargs else None
        return Clustering_predictor(training_data=train_data, idx2comb=kwargs["idx2comb"], features=kwargs["features"], hyperparameters=hyperparameters) 
    elif predictor_type == "autofolio":
        if "features" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs features. features cannot be None")
        max_threads = kwargs["max_threads"] if "max_threads" in kwargs else 12
        pre_trained_model = kwargs["pre_trained_model"] if "pre_trained_model" in kwargs else None
        return Autofolio_predictor(training_data=train_data, features=kwargs["features"], fold=kwargs["fold"], max_threads=max_threads, pre_trained_model=pre_trained_model)
    elif predictor_type == "metric":
        if "features" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs features. features cannot be None")
        if "idx2comb" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs idx2comb. idx2comb cannot be None")
        if "metrics_type" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs a metric type. metrics_type cannot be None")
        return Metrics_predictor(training_data=train_data, idx2comb=kwargs["idx2comb"], features=kwargs["features"], metrics_type=kwargs["metrics_type"])
    else:
        raise Exception(f"predictor_type {predictor_type} unrecognised")

def get_dataloader(x, y, test_buckets = []):
    BUCKETS = 10

    N_ELEMENTS = len(x)

    BUCKET_SIZE = N_ELEMENTS // BUCKETS

    x_local = x.copy()
    y_local = y.copy()
    x_test, y_test = [], []

    for bucket in test_buckets:
        idx = bucket * BUCKET_SIZE
        for _ in range(BUCKET_SIZE):
            x_test.append(x_local.pop(idx))
            y_test.append(y_local.pop(idx))

    train_elements = (len(y_local) // 10) * 9
    x_train = x_local[:train_elements]
    y_train = y_local[:train_elements]

    x_validation = x_local[train_elements:]
    y_validation = y_local[train_elements:]
    
    return  (x_train, y_train), (x_validation, y_validation), (x_test, y_test)

def is_competitive(vb, option):
        return (option < 10 or vb * 2 >= option) and option < 3600

def get_sb_vb(train:'list[dict]', validation:'list[dict]', test:'list[dict]') -> 'tuple[tuple[float,float],tuple[float,float],tuple[float,float]]':
    sb_combination = "chuffed_02_compact.eprime"
    sb_train, sb_val, sb_test = 0, 0, 0
    vb_train, vb_val, vb_test = 0, 0, 0

    for datapoint in train:
        vb_train += datapoint["time"]
        for t in datapoint["all_times"]:
            if t["combination"] == sb_combination:
                sb_train += t["time"]
                break

    for datapoint in validation:
        vb_val += datapoint["time"]
        for t in datapoint["all_times"]:
            if t["combination"] == sb_combination:
                sb_val += t["time"]
                break

    for datapoint in test:
        vb_test += datapoint["time"]
        for t in datapoint["all_times"]:
            if t["combination"] == sb_combination:
                sb_test += t["time"]
                break
    return (sb_train, vb_train), (sb_val, vb_val), (sb_test, vb_test)

def positive_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def pad(*args) -> tuple:
    str_args = [str(arg) for arg in args]
    max_len_arg = max([len(arg) for arg in str_args])
    str_args = [f"{' '* (max_len_arg - len(arg))}{arg}" for arg in str_args]
    return tuple(str_args)