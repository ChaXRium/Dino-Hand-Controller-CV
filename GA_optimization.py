import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Load Data (Must be defined before the loop!)
img_size = (64, 64)
batch_size = 32

print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary'
)

# 2. The "Genetic" Search Space
learning_rates = [0.01, 0.001, 0.0001]
filter_sizes = [16, 32, 64]

def create_individual(lr, filters):
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(filters, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model
