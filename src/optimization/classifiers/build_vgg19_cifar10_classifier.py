import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

import numpy as np
import matplotlib.pyplot as plt
import os.path
import argparse
import time


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.applications.VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3),
    classifier_activation='relu'
)

start_time = time.time()
_, base_model_accuracy = model.evaluate(test_images, test_labels, verbose=1)

batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.
num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_for_pruning.summary()
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
_, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images, test_labels, verbose=1)
print("\nBase model's accuracy: ", base_model_accuracy)
print("Pruned model's accuracy: ", model_for_pruning_accuracy)

# predictions = model.predict(test_images)
end_time = time.time()
print(f"\nTime for prediction of {len(test_labels)} images: {end_time - start_time} seconds")

model.save('vgg19_cifar10_classifier.h5')
