import pandas as pd


class BasePredictor:
    def __init__(self, variable_names: list or str, clean_data=False):
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        self.data = {name: pd.read_csv("data/forecasts/%s.csv" % name) for name in variable_names}
        if clean_data:
            self._clean_data()

        self.predictions = None
        self.actual_values = None

    def _clean_data(self):
        pass

    def _load_actual_values(self):
        self.actual_values = {}

        for var, pred in self.predictions.items():
            actual = pd.read_csv("data/actual/%s.csv" % var).set_index("DATE")
            for c in pred.columns:
                h = int(c[-1])
                actual["A%s%i" % (var, h)] = actual[var].shift(2 - h)
            actual = actual.drop(columns=[var])
            self.actual_values[var] = actual

    def calculate_metrics(self):
        if self.actual_values is None:
            self._load_actual_values()

        mse = {}
        for var, pred in self.predictions.items():
            aux = self.actual_values[var].reindex(pred.index)
            aux.columns = pred.columns
            mse[var] = pred.sub(aux).pow(2).mean().pow(0.5)

        return mse

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement fit")


class AverageForecasts(BasePredictor):
    def __init__(self, variable_names: list or str, forecast_horizons: list, clean_data=False):
        super().__init__(variable_names, clean_data)
        self.forecast_horizons = forecast_horizons

    def fit(self):
        self.predictions = {}
        for var in self.data.keys():
            pred = self.data[var].groupby(["DATE"]).mean()
            cols = [var + str(h+2) for h in self.forecast_horizons]
            self.predictions[var] = pred.loc[:, cols]
        return self


class MedianForecasts(BasePredictor):
    def __init__(self, variable_names: list or str, forecast_horizons: list, clean_data=False):
        super().__init__(variable_names, clean_data)
        self.forecast_horizons = forecast_horizons

    def fit(self):
        self.predictions = {}
        for var in self.data.keys():
            pred = self.data[var].groupby(["DATE"]).median()
            cols = [var + str(h+2) for h in self.forecast_horizons]
            self.predictions[var] = pred.loc[:, cols]
        return self

