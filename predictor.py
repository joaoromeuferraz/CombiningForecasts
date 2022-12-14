import pandas as pd
import numpy as np
import pickle
import os

np.seterr(divide='ignore')


class BasePredictor:
    def __init__(self, variable_names: list or str, forecast_horizons: list, lb_cov=36, min_periods=4,
                 clean_data=False, data=None):
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        if data is None:
            self.data = {name: pd.read_csv("data/forecasts/%s.csv" % name) for name in variable_names}
            self.using_file_data = True
        else:
            self.data = data
            self.using_file_data = False

        if clean_data:
            self._clean_data()

        self.forecast_horizons = forecast_horizons
        self.predictions = None
        self.actual_values = None
        self.lb_cov = lb_cov
        self.min_periods = min_periods
        self.cov = None
        self.data_pivot = None
        self.pred_errors = None

    def _clean_data(self):
        pass

    def _load_actual_values(self):
        self.actual_values = {}

        for var in self.data.keys():
            actual = pd.read_csv("data/actual/%s.csv" % var).set_index("DATE")
            for h in self.forecast_horizons:
                h = int(h + 2)
                actual["A%s%i" % (var, h)] = actual[var].shift(2 - h)
            actual = actual.drop(columns=[var])
            self.actual_values[var] = actual

    def calculate_metrics(self, start_date=None, end_date=None):
        if self.actual_values is None:
            self._load_actual_values()

        mse = {}
        mad = {}
        for var, pred in self.predictions.items():
            aux = self.actual_values[var].reindex(pred.index)
            if start_date is not None:
                aux = aux.loc[start_date:]
            if end_date is not None:
                aux = aux.loc[:end_date]
            aux.columns = pred.columns
            mse[var] = pred.sub(aux).pow(2).mean().pow(0.5)
            mad[var] = np.abs(pred.sub(aux)).mean()

        return mse, mad

    def estimate_cov(self, rewrite=False):
        if os.path.exists(f"data/cov_data_{self.lb_cov}.p") and not rewrite and self.using_file_data:
            self.cov, self.data_pivot, self.pred_errors = pickle.load(open(f"data/cov_data_{self.lb_cov}.p", "rb"))
            return None

        if self.actual_values is None:
            self._load_actual_values()

        res = {}
        data = {}
        pred_errors = {}
        for var in self.data.keys():
            res[var] = {}
            data[var] = {}
            pred_errors[var] = {}
            for h in self.forecast_horizons:
                data_pivot = self.data[var].pivot_table(f"{var}{h + 2}", index="DATE", columns="ID")
                data[var][h] = data_pivot
                pred_error = data_pivot.sub(self.actual_values[var][f"A{var}{h + 2}"], axis=0).dropna(how='all')
                pred_errors[var][h] = pred_error

                start_dt = pred_error.index[self.lb_cov]
                cov = pd.DataFrame(dtype=float, index=pred_error.index, columns=pred_error.columns)
                for i, dt in enumerate(pred_error.index):
                    if dt >= start_dt:
                        aux_cov = pd.Series(dtype=float, index=pred_error.columns, name=dt)
                        for p_id in pred_error.columns:
                            aux_series = pred_error.iloc[:i - 1][p_id].dropna()
                            if len(aux_series) >= self.min_periods:
                                aux_cov[p_id] = aux_series.var()
                            else:
                                aux_cov[p_id] = np.nan

                    else:
                        aux_cov = pd.Series(np.nan, index=pred_error.columns, name=dt)
                    cov.loc[dt] = aux_cov
                res[var][h] = cov

        self.cov = res
        self.data_pivot = data
        self.pred_errors = pred_errors

        if self.using_file_data:
            pickle.dump([self.cov, self.data_pivot, self.pred_errors], open(f"data/cov_data{self.lb_cov}", "wb"))

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement fit")


class AverageForecasts(BasePredictor):
    def __init__(self, variable_names: list or str, forecast_horizons: list, clean_data=False):
        super().__init__(variable_names, forecast_horizons, clean_data=clean_data)

    def fit(self):
        self.predictions = {}
        for var in self.data.keys():
            pred = self.data[var].groupby(["DATE"]).mean()
            cols = [var + str(h + 2) for h in self.forecast_horizons]
            self.predictions[var] = pred.loc[:, cols]
        return self


class MedianForecasts(BasePredictor):
    def __init__(self, variable_names: list or str, forecast_horizons: list, clean_data=False):
        super().__init__(variable_names, forecast_horizons, clean_data=clean_data)
        self.forecast_horizons = forecast_horizons

    def fit(self):
        self.predictions = {}
        for var in self.data.keys():
            pred = self.data[var].groupby(["DATE"]).median()
            cols = [var + str(h + 2) for h in self.forecast_horizons]
            self.predictions[var] = pred.loc[:, cols]
        return self


class GeomAverageForecasts(BasePredictor):

    def __init__(self, variable_names: list or str, forecast_horizons: list, clean_data=False):
        super().__init__(variable_names, forecast_horizons, clean_data=clean_data)

    def fit(self):
        self.predictions = {}
        for var in self.data.keys():
            col_names = self.data[var].columns.tolist()[1:]
            pred = pd.DataFrame(self.data[var]).groupby(["DATE"])[col_names].apply(lambda x: np.exp((np.log(x)).mean()))
            # print(pred)
            cols = [var + str(h + 2) for h in self.forecast_horizons]
            self.predictions[var] = pred.loc[:, cols]
        return self


class HarmAverageForecasts(BasePredictor):

    def __init__(self, variable_names: list or str, forecast_horizons: list, clean_data=False):
        super().__init__(variable_names, forecast_horizons, clean_data=clean_data)

    def fit(self):
        self.predictions = {}
        for var in self.data.keys():
            col_names = self.data[var].columns.tolist()[1:]
            pred = pd.DataFrame(self.data[var]).groupby(["DATE"])[col_names].apply(lambda x: 1 / ((1 / x).mean()))
            # print(pred)
            cols = [var + str(h + 2) for h in self.forecast_horizons]
            self.predictions[var] = pred.loc[:, cols]
        return self


class AverageMedianForecasts(BasePredictor):

    def __init__(self, variable_names: list or str, forecast_horizons: list, clean_data=False):
        super().__init__(variable_names, forecast_horizons, clean_data=clean_data)

    def fit(self):
        self.predictions = {}
        for var in self.data.keys():
            col_names = self.data[var].columns.tolist()[1:]
            pred = pd.DataFrame(self.data[var]).groupby(["DATE"])[col_names].apply(
                lambda x: (x.mean() + x.median()) / 2)
            # print(pred)
            cols = [var + str(h + 2) for h in self.forecast_horizons]
            self.predictions[var] = pred.loc[:, cols]
        return self


class NormalIndependent(BasePredictor):
    def __init__(self, variable_names: list or str, forecast_horizons: list, lb_cov=36, min_periods=4,
                 clean_data=False):
        super().__init__(variable_names, forecast_horizons, lb_cov, min_periods, clean_data)
        self._load_actual_values()

    def fit(self):
        self.estimate_cov()
        cov = self.cov
        self.predictions = {}
        for var in cov.keys():
            dates = cov[var][self.forecast_horizons[0]].index
            pred = pd.DataFrame(dtype=float, index=dates, columns=[f"{var}{h + 2}" for h in self.forecast_horizons])
            for h in cov[var].keys():
                col = f"{var}{h + 2}"
                for dt in cov[var][h].index:
                    cov_inv = np.reciprocal(cov[var][h].loc[dt])
                    aux = pd.concat((cov_inv, self.data_pivot[var][h].loc[dt]), axis=1, keys=["s", "p"]).dropna()
                    aux["s"] /= aux["s"].sum()
                    pred.loc[dt, col] = aux["s"] @ aux["p"]
            self.predictions[var] = pred
        return self


class BayesianModels(BasePredictor):
    def __init__(self, variable_names: list or str, forecast_horizons: list, lb_cov=36, min_periods=4,
                 alpha=36, clean_data=False):
        super().__init__(variable_names, forecast_horizons, lb_cov, min_periods, clean_data)
        self._load_actual_values()
        self.alpha = alpha

    def fit(self):
        self.estimate_cov()
        cov = self.cov
        self.predictions = {}
        for var in cov.keys():
            dates = cov[var][0].index
            pred = pd.DataFrame(dtype=float, index=dates, columns=[f"{var}{h + 2}" for h in self.forecast_horizons])
            for h in cov[var].keys():
                dates = list(cov[var][h].index)
                prior = cov[var][h].loc[dates[0]]
                prior_inv = np.reciprocal(prior)
                col = f"{var}{h + 2}"
                for dt in cov[var][h].index:
                    sample_inv = np.reciprocal(cov[var][h].loc[dt])
                    cov_inv = (self.alpha * prior_inv + self.lb_cov * sample_inv) / (self.alpha + self.lb_cov)
                    aux = pd.concat((cov_inv, self.data_pivot[var][h].loc[dt]), axis=1, keys=["s", "p"]).dropna()
                    aux["s"] /= aux["s"].sum()
                    pred.loc[dt, col] = aux["s"] @ aux["p"]
                    prior_inv = sample_inv
            self.predictions[var] = pred
        return self


class OnlineLearning(BasePredictor):
    def __init__(self, variable_names: list or str, forecast_horizons: list, alpha=0., lb_cov=36, min_periods=4,
                 clean_data=False, data=None):
        super().__init__(variable_names, forecast_horizons, lb_cov, min_periods, clean_data, data)
        self._load_actual_values()
        self.alpha = alpha
        self.weights = None

    def fit(self):
        self.estimate_cov()
        cov = {var: {h: x.dropna(how='all') for h, x in self.cov[var].items()} for var in self.cov.keys()}
        self.predictions = {}
        self.weights = {}
        for var in cov.keys():
            dates = list(cov[var][0].index)
            pred = pd.DataFrame(dtype=float, index=dates, columns=[f"{var}{h + 2}" for h in self.forecast_horizons])
            pred.index.name = "DATE"
            ws = {}
            for h in cov[var].keys():
                w_df = pd.DataFrame(dtype=float, index=dates, columns=cov[var][h].columns)
                w = np.repeat(1. / len(cov[var][h].columns), len(cov[var][h].columns))
                w = pd.Series(w, index=cov[var][h].columns)
                dates = list(cov[var][h].index)
                col = f"{var}{h + 2}"
                for i, dt in enumerate(cov[var][h].index):
                    mse = self.pred_errors[var][h].loc[:dt].pow(2).dropna(thresh=self.min_periods, axis=1).mean(axis=0)
                    mse_norm = (mse - mse.mean()) / (mse.std() + 1e-8)
                    target = np.exp(-mse_norm) / np.exp(-mse_norm).sum()
                    w_old = w.reindex(target.index, method=None)
                    w_old[pd.isna(w_old)] = np.mean(w_old)
                    w_new = w_old * target / (w_old * target).sum()
                    preds = self.data_pivot[var][h].loc[dt].reindex(w_new.index, method=None).dropna()
                    w_new = w_new.reindex(preds.index, method=None)
                    w_new = w_new.dropna()
                    w_new /= w_new.sum()
                    w_old = w_old.reindex(w_new.index, method=None)
                    w_old[pd.isna(w_old)] = np.mean(w_old)
                    w_old /= w_old.sum()

                    w_new = (1-self.alpha)*w_new + self.alpha*w_old
                    pred.loc[dt, col] = w_new @ preds

                    w_df.loc[dt, :] = w_new.reindex(index=cov[var][h].columns, method=None)
                    w = w_new
                ws[h] = w_df
            self.weights[var] = ws
            self.predictions[var] = pred
        return self
