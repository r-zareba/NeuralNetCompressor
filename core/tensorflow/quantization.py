from core import tensorflow as tf
import numpy as np


def int8_quantization(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    return converter.convert()


def full_int8_quantization(model: tf.keras.Model,
                           representative_dataset: np.array,
                           force=False) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = ([representative_dataset[i]]
                                        for i in range(len(representative_dataset)))

    if force:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    return converter.convert()


def float16_quantization(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()
