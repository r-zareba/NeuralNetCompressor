from core import tensorflow as tf
import numpy as np
from typing import Tuple


# TODO Evaluator for TFLite models

class TFLiteRunner:
    """
    Implementation of efficient model inference using tflite models
    """
    __slots__ = ('_interpreter', '_input_idx', '_output_idx', '_input_dtype',
                 '_input_size')

    def __init__(self, tflite_model: bytes = None, model_path='', input_size=()):
        if not tflite_model and not model_path:
            raise ValueError('Cannot initialize TFLiteRunner with '
                             'no bytes model or path to it !')
        if tflite_model:
            self._interpreter = tf.lite.Interpreter(model_content=tflite_model)
        elif model_path:
            self._interpreter = tf.lite.Interpreter(model_path=model_path)

        self._input_idx = self._interpreter.get_input_details()[0]['index']
        self._output_idx = self._interpreter.get_output_details()[0]['index']
        self._input_dtype = self._interpreter.get_input_details()[0]['dtype']
        self._input_size = ()

    @property
    def input_size(self) -> Tuple:
        return self._input_size

    @input_size.setter
    def input_size(self, input_size: Tuple):
        self._input_size = input_size
        self._resize_input_tensor(input_size)

    def inference(self, x: np.array) -> np.array:
        if x.shape != self._input_size:
            self._resize_input_tensor(x.shape)
        return self.fast_inference(x)

    def fast_inference(self, x: np.array) -> np.array:
        self._interpreter.set_tensor(
            self._input_idx, x.astype(self._input_dtype))
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_idx)

    def _resize_input_tensor(self, shape: Tuple) -> None:
        self._interpreter.resize_tensor_input(
            input_index=self._input_idx, tensor_size=shape)
        self._interpreter.allocate_tensors()
