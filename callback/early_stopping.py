# import torch
#
# from model import BaseModel
# from callback import Callback
# from torch.utils.data import Dataset
#
#
# class EarlyStopping(Callback):
#
#     def __init__(self,
#                  model: BaseModel,
#                  validation_set: Dataset,
#                  monitor: str = 'loss',
#                  patience: int = 3,
#                  keep_best_model: bool = True,
#                  precision: int = 3):
#         """
#         :param validation_set: Dataset to monitor
#         :param monitor: measure to monitor
#         :param patience: number of times to wait until stop if the monitored measure is getting worse
#         :param keep_best_model: whether we must save the best parameters and update the model with them at the end
#         :param precision: how many decimal places must be used to check for improvement
#         """
#
#         super().__init__()
#
#         self._model = model
#         self._validation_set = validation_set
#         self._monitor = monitor
#         self._patience = patience
#         self._keep_best_model = keep_best_model
#         self._precision = precision
#
#         self._last_measure = None
#         self._non_improvement_counter = 0
#
#     def on_training_begin(self, train: bool):
#         self._last_measure = None
#         self._non_improvement_counter = 0
#
#     def _on_train_epoch_end(self, epoch: int, logs: Dict[str, float], train: bool):
#         with torch.no_grad():
#             measure = 0
#             if self._monitor == 'loss':
#                 self._model.eval()
#                 # measure = self._model.calculate_loss(self._validation_set.)
#                 self._model.train()
#
#             measure = round(measure, self._precision)
#
#             if self._last_measure is None or measure < self._last_measure:
#                 self._model.stash_parameters()
#                 self._non_improvement_counter = 0
#             else:
#                 self._non_improvement_counter += 1
#
#             if self._non_improvement_counter > self._patience:
#                 print('Early stopping.')
#                 self._model.stop_training = True
#
#                 if self._keep_best_model:
#                     self._model.pop_parameters()
#
#             self._last_measure = measure
