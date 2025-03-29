# Import Necessary Packages
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import Dataset(s)
train_path = 'Insert Path Here'
test_path = 'Insert Path Here'

train=pd.read_csv(train_path)
test=pd.read_csv(test_path)

print(f"There are {train.shape[0]} images in the training data")
print(f"There are {test.shape[0]} images in the test data")

# Simple Visualizations From Datasets
# Distribution of Digits in Train Dataset
plt.figure(figsize=(16, 6))
sns.countplot(data = train, x = 'label')
plt.ylabel('Frequency');

# Images of Digits in Dataset
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(train.iloc[i, 1:].values.reshape(28, 28))
    ax.set_title(f"Label: {train.iloc[i, 0]}")
    ax.axis('off')

# Normalize Images for Uniformity
X_train=train.drop(columns=['label']).values.reshape(train.shape[0],28,28,1)/255.0
y_train=train['label']
X_test=test.values.reshape(test.shape[0],28,28,1)/255.0

# Split Training Data into Training and Validation
X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 420)

# Convoluted Neural Network (CNN) Model (2 Convoluted and 2 Pooling Layers)
model=Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
    input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.summary()

# Compile CNN Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate Model Performance on Training Data
y_train_pred = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred, axis=1)

class_report = classification_report(y_train, y_train_pred)#, target_names=[str(i) for i in range(10)])
print(class_report)

# Visualize Classification Report
conf_matrix = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Evaluate Model Performance on Validation Data
y_val_pred = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred, axis=1)

class_report = classification_report(y_val, y_val_pred)#, target_names=[str(i) for i in range(10)])
print(class_report)

# Visualize Classification Report
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Kaggle Competition Submission
# Use Model to Make Predictions on Test Data
predictions=model.predict(X_test)
predictions=np.argmax(predictions,axis=1)

# Export Predictions and Create Labels (Kaggle submission scored 98.52% accuracy)
prediction = pd.DataFrame(predictions, columns=['Label'])
ids = pd.DataFrame(range(1,predictions.shape[0]+1),columns=['ImageId'])
output = pd.concat([ids, prediction],axis = 1)
output.to_csv('submission.csv',index=False)