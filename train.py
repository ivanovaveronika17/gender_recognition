import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path


# 1. Поставки

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

project_folder = Path(__file__).parent

train_folder = project_folder / "dataset" / "train"
validation_folder = project_folder / "dataset" / "validation"


# 2. Ги вчитуваме сликите

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    labels="inferred",
    label_mode="binary",
    class_names=["male", "female"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_folder,
    labels="inferred",
    label_mode="binary",
    class_names=["male", "female"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 3. Нормализација

normalization_layer = layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)


# 4. Го земаме MobileNetV2

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# 5. Не ги менуваме почетните слоеви

base_model.trainable = False


# 6. Го додаваме нашиот classification дел

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])


# 7. Го компајлираме моделот

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# 8. Го прикажуваме моделот

model.summary()


# 9. Го тренираме моделот

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)


# 10. Го зачувуваме моделот

model.save("models/gender_model.keras")

print()
print("Моделот е успешно трениран!")
print("Зачуван е во: models/gender_model.keras")