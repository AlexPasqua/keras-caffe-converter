import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

import numpy as np
import matplotlib.pyplot as plt
import os.path
import argparse
import time


# Images
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Model
model = tf.keras.applications.VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3),
    classifier_activation='relu'
)

# Training hyperparameters
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.
num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# compile (necesasry for model.evaluate())
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# No training because I loaded pretrained weights
# evaluate
print('Base model evaluation')
_, base_model_accuracy = model.evaluate(test_images, test_labels, verbose=1)


# PRUNING
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

pruned_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

pruned_model.summary()

callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

print('Fit of pruned model')
pruned_model.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

print('Pruned model evaluation')
_, pruned_model_accuracy = pruned_model.evaluate(test_images, test_labels, verbose=1)

print("\nBase model's accuracy: ", base_model_accuracy)
print("Pruned model's accuracy: ", pruned_model_accuracy)

# predictions = model.predict(test_images)
#end_time = time.time()
#print(f"\nTime for prediction of {len(test_labels)} images: {end_time - start_time} seconds")

#model.save('vgg19_cifar10_classifier.h5')
