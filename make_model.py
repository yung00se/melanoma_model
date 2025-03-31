import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv

with open('model_parameters.json', 'r') as f:
    params = json.load(f)

version = params['version']
benign_images = params['benign_images']
malignant_images = params['malignant_images']
convolutional_layers = params['convolutional_layers']
dense_layers = params['dense_layers']
dense_nodes = params['dense_nodes']
batch_size = params['batch_size']
dropout = params['dropout']
epochs = params['epochs']
threshold = params['threshold']

data_dir = 'melanoma_cancer_dataset'
training_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

train = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    label_mode='binary',
    image_size=(256, 256),
    shuffle=True
)

val = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    label_mode='binary',
    image_size=(256, 256),
    shuffle=True
)

test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='binary',
    image_size=(256, 256),  # Resize to match model input
    batch_size=32,
    shuffle=False  # Keep order to match predictions with actual labels
)

class_names = test.class_names

train = train.map(lambda x,y: (x/255.0, y))
val = val.map(lambda x,y: (x/255.0, y))
test = test.map(lambda x,y: (x/255.0, y))

layers = [
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D()
]

nodes = 64
for i in range(1, convolutional_layers):
    layers.append(Conv2D(nodes, (3, 3), activation='relu'))
    layers.append(BatchNormalization())
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    nodes *= 2

layers.append(Flatten())

for i in range(dense_layers):
    layers.append(Dense(dense_nodes, activation='relu'))

layers.append(Dropout(dropout))
layers.append(Dense(1, activation='sigmoid'))

model = Sequential(layers)

model.summary()
print("Version:", version)
print("Benign training images:", len(os.listdir("melanoma_cancer_dataset/train/benign")))
print("Malignant training images:", len(os.listdir("melanoma_cancer_dataset/train/malignant")))
print("Benign validation images:", len(os.listdir("melanoma_cancer_dataset/validation/benign")))
print("Malignant validation images:", len(os.listdir("melanoma_cancer_dataset/validation/malignant")))
print("Batch Size:", batch_size)
print("Dropout Rate:", dropout)
print("Epochs:", epochs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history = model.fit(
    train,
    epochs=epochs,
    validation_data=val
)

# Plotting training history
def plot_metrics(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    for metric in metrics:
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.show()

def log_metrics(history, conf_matrix):
    true_negatives = conf_matrix[0][0]
    false_positives = conf_matrix[0][1]
    false_negatives = conf_matrix[1][0]
    true_positives = conf_matrix[1][1]

    total = true_negatives + false_positives + false_negatives + true_positives
    accuracy = (true_negatives + true_positives) / total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    with open('progress_log.csv', 'r') as r:
        version = r.readlines()[-1].split(',')[0]
        tens, tenths = version[0], version[1]
        print(tens, tenths)
        #version = version.split('.')[0] + '.' + str(int(version.split('.')[1]) + 1)


    new_iteration = [
        version,
        benign_images,
        malignant_images,
        convolutional_layers,
        dense_layers,
        dense_nodes,
        batch_size,
        dropout,
        epochs,
        threshold,
        accuracy,
        precision,
        recall,
        f1_score
    ]
    
    with open('progress_log.csv', mode='a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(new_iteration)

plot_metrics(history)

# Save the trained model
model.save('melanoma_detector.keras')

# Load the trained model
model = tf.keras.models.load_model("melanoma_detector.keras")

# Make predictions
y_pred_prob = model.predict(test)  # Get probabilities
y_pred = (y_pred_prob > threshold).astype(int)  # Convert probabilities to binary labels
#print(y_pred)

# Get true labels from the dataset
y_true = np.concatenate([y for x, y in test], axis=0)

# Evaluate performance
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
log_metrics(history, conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

