# Pet Classification

# Import Necessary Libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16 ,Xception

# Import Datasets for Training and Testing Model
# Define the directory paths for the training and validation datasets
train_dir = 'Insert Path Here'
test_dir = 'Insert Path Here'

# Specify Image and Batch Parameters
batch_size = 32
img_height = 256
img_width = 256
seed = 420

# Define the directory paths for the training and validation datasets
train_dir = '/kaggle/input/dogs-vs-cats/train'
test_dir = '/kaggle/input/dogs-vs-cats/test'

# Use the image_dataset_from_directory function to create training and validation dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.25,
    subset="training",
    class_names=None,
    shuffle = True,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    seed=seed)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.25,
    subset= "validation",
    color_mode="rgb",
    class_names=None,
    shuffle = True,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=seed)

# Use the image_dataset_from_directory function to create test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=test_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    seed=seed)

# Standardize Images in the Dataset
def image_norm (image,label):
    image = tf.cast(image/255,tf.float32)
    return image,label

train_ds = train_ds.map(image_norm)
test_ds = test_ds.map(image_norm)
val_ds = val_ds.map(image_norm)

# To clear any previously fitted models
keras.backend.clear_session()

# CNN Model 1 (No Pre-Trained Base Model)
model = Sequential()
model.add(Convolution2D(filters = 32, kernel_size = (5,5), activation = "relu",
                        input_shape = (256,256,3)))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Convolution2D(filters = 32, kernel_size = (5,5), activation = "relu"))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Convolution2D(filters = 32, kernel_size = (5,5), activation = "relu"))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Flatten())
model.add(Dense(units = 128, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(units = 2, activation = "sigmoid"))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs= 10,
                    validation_data=val_ds,
                    callbacks= EarlyStopping(min_delta=0.001, patience=5, restore_best_weights=True,))

# Accuracy - Epoch Plot
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='test')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Model Accuracy - Self-Trained Model')
plt.legend()
plt.show()

# Loss-Epoch Plot
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='test')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Model Loss - Self-Trained Model')
plt.legend()
plt.show()

# Test Data on Unseen Test Dataset
loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Test loss: {loss:.2f}")

# To clear any previously fitted models
keras.backend.clear_session()

# CNN Model 2 (With Pre-Trained Base Model)
# Instantiate Pre-Trained Image Classification Model
conv_base = Xception(
    weights='imagenet',
    include_top = False,
    input_shape =(img_height, img_width, 3),
    pooling='avg')

conv_base.trainable = False

# Build Model, Incorporating the Pre-Trained Image Classification Model
model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(Dense(120, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))

# Compile Built Model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit Model on Training Data and Check its Accuracy with Unseen Test Data
history = model.fit(train_ds,
                    epochs= 10,
                    validation_data = val_ds,
                    callbacks = EarlyStopping(min_delta = 0.001, patience = 5, restore_best_weights = True,))

# Accuracy - Epoch Plot
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='test')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Model Accuracy - With Pre-Trained Model')
plt.legend()
plt.show()

# Loss-Epoch Plot
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='test')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Model Loss - With Pre-Trained Model')
plt.legend()
plt.show()

# Test Data on Unseen Test Dataset
loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Test loss: {loss:.2f}")