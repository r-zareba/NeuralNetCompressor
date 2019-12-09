import tensorflow as tf
import tensorflow_model_optimization.sparsity.keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
import tensorflow.keras.backend as K
import numpy as np


def _get_sparsity(weights: np.array):
    return 1.0 - np.count_nonzero(weights) / float(weights.size)


# def print_model_sparsity(pruned_model) -> None:
#     for layer in pruned_model.layers:
#         if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
#             prunable_weights = layer.layer.get_prunable_weights()
#             for weight in prunable_weights:
#                 print(str(_get_sparsity(K.get_value(weight))))


def print_model_sparsity(model):
    for layer in model.layers:
        if layer.weights:
            print(_get_sparsity(K.get_value(layer.weights[0])))


def calculate_last_train_step(train_dataset: np.array, batch_size: int,
                              n_epochs: int) -> int:
    n_samples = train_dataset.shape[0]
    return np.ceil(1.0 * n_samples / batch_size).astype(np.int32) * n_epochs


class Pruner:
    """
    Implementation of Tensorflow whole model pruner
    Affects all weights during pruning
    """
    __slots__ = ('_model', '_initial_sparsity', '_begin_step',
                 '_final_sparsity', '_frequency')

    def __init__(self, model: tf.keras.Model, initial_sparsity=0.3, begin_step=0,
                 final_sparsity=0.8, frequency=100) -> None:
        self._model = model
        self._initial_sparsity = initial_sparsity
        self._begin_step = begin_step
        self._final_sparsity = final_sparsity
        self._frequency = frequency

    def compile(self, x_train: np.array, batch_size: int, n_epochs: int,
                **compile_kwargs) -> None:
        end_step = calculate_last_train_step(x_train, batch_size, n_epochs)

        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=self._initial_sparsity,
                final_sparsity=self._final_sparsity,
                begin_step=self._begin_step,
                end_step=end_step,
                frequency=self._frequency)
        }
        self._model = sparsity.prune_low_magnitude(self._model, **pruning_params)
        self._model.compile(**compile_kwargs)

    def fit(self, **fit_params):
        if 'callbacks' in fit_params.keys():
            fit_params['callbacks'].append(sparsity.UpdatePruningStep())
        else:
            fit_params['callbacks'] = [sparsity.UpdatePruningStep()]

        return self._model.fit(**fit_params)

    def get_model(self) -> tf.keras.Model:
        return sparsity.strip_pruning(self._model)

