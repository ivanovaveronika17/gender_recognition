import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from pathlib import Path

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

project_folder = Path(__file__).parent

test_folder = project_folder / "dataset" / "test"
model_path = project_folder / "models" / "gender_model.keras"

model = tf.keras.models.load_model(model_path)

print("Моделот е успешно вчитан!")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    labels="inferred",
    label_mode="binary",
    class_names=["male", "female"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

true_labels = []

for images, labels in test_dataset:
    true_labels.extend(labels.numpy().flatten())

true_labels = np.array(true_labels).astype(int)

predictions = model.predict(test_dataset)

predicted_labels = (predictions >= 0.5).astype(int).flatten()

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels
)

recall = recall_score(
    true_labels,
    predicted_labels
)

print()
print("===== RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=["male", "female"]
    )
)
