import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

import core.tensorflow.pruning as pruning

# Download the CIFAR dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 64
n_epochs = 8

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# start_time = time.time()
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(train_images, train_labels, batch_size=batch_size,
#                     epochs=10, validation_data=(test_images, test_labels))
#
# test_loss, test_acc = model.evaluate(test_images,  test_labels)

pruner = pruning.Pruner(model=model)
pruner.compile(x_train=train_images, batch_size=batch_size, n_epochs=n_epochs,
               optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

pruner.fit(x=train_images, y=train_labels, batch_size=batch_size,
           epochs=n_epochs, validation_data=(test_images, test_labels))


sparse_model = pruner.get_model()
pruning.print_model_sparsity(sparse_model)