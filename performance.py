import pandas as pd

#
# class Performance:
#     def __init__(self, predictions):
#         self.predictions = predictions
#         self.actual_values = {}
#
#         for var, pred in self.predictions.items():
#             actual = pd.read_csv("data/actual/%s.csv" % var).set_index("DATE")
#             for c in pred.columns:
#                 h = int(c[-1])
#                 actual["A%s%i" % (var, h)] = actual[var].shift(2-h)
#             actual = actual.drop(columns=[var])
#             self.actual_values[var] = actual
#
#     def calculate_metrics(self):
#         mse = {}
#         for var, pred in self.predictions.items():
#             aux = self.actual_values[var].reindex(pred.index)
#             aux.columns = pred.columns
#             mse[var] = pred.sub(aux).pow(2).mean().pow(0.5)
#
#         return mse
#
#
#
#
