import tensorflow
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 256, 256
train_data_dir = "/storage/imagenette2/train/"
validation_data_dir = "/storage/imagenette2/val/"
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 50

model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))



# Freeze the layers which you don't want to train. Here I am freezing the all layers.
for layer in model.layers[:]:
    layer.trainable = False

# Adding custom Layer
# We only add
x = model.output
x = Flatten()(x)
# Adding even more custom layers
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

# creating the final model
model_final = Model(inputs = model.input, outputs = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range=0.3,
  rotation_range=30)

test_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range=0.3,
  rotation_range=30)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size = (img_height, img_width),
  batch_size = batch_size,
  class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
  validation_data_dir,
  target_size = (img_height, img_width),
  class_mode = "categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("resnet50_retrain.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model
model_final.fit_generator(
  train_generator,
  ##steps_per_epoch = nb_train_samples,
  epochs = epochs,
  validation_data = validation_generator,
  #validation_steps = nb_validation_samples
  #callbacks = [checkpoint, early]
)
