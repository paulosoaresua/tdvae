from typing import Any, Dict


class Callback:
    def __init__(self):
        pass

    def on_training_begin(self, model: Any):
        pass

    def on_train_batch_begin(self, batch: int):
        pass

    def on_train_batch_end(self, batch: int, logs: Dict[str, float]):
        pass

    def on_train_epoch_begin(self, epoch: int):
        pass

    def on_train_epoch_end(self, epoch: int, logs: Dict[str, float]):
        pass
