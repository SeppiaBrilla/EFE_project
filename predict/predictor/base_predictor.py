class Predictor_initializer:
    pass

class Predictor:
    def predict(self, dataset:'list[dict]|list[float]', filter:'bool'=False) -> 'list[dict]|str':
        raise Exception("Not implemented method")

    def __get_dataset(self, dataset:'list') -> 'list[dict]':
        if type(dataset[0]) == float:
            return [{"inst":"", "features":dataset}]
        return dataset


