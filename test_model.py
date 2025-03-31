import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define the path to your test data directory
test_dir = "melanoma_cancer_dataset/test"
#test_dir = "test"

# Load the test dataset with automatic labeling
test = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='binary',
    image_size=(256, 256),  # Resize to match model input
    batch_size=32,
    shuffle=False  # Keep order to match predictions with actual labels
)

# Extract class names (should be ["benign", "malignant"])
class_names = test.class_names
#class_names.reverse()
print("Class Labels:", class_names)  # Output: ['benign', 'malignant']

test = test.map(lambda x,y: (x/255.0, y))

# Load the trained model
model = tf.keras.models.load_model("melanoma_detector.keras")

# Make predictions
y_pred_prob = model.predict(test)  # Get probabilities

y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels
#print(y_pred)

# Get true labels from the dataset
y_true = np.concatenate([y for x, y in test], axis=0)

# Evaluate performance
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

