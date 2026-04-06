import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# 1. Load Data with Augmentation (Advanced Data Integrity)
# This creates 'fake' variations of your images to make the model harder to fool
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset', validation_split=0.2, subset="training", seed=123,
    image_size=(128, 128), batch_size=32, label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset', validation_split=0.2, subset="validation", seed=123,
    image_size=(128, 128), batch_size=32, label_mode='binary'
)

# 2. Use MobileNetV2 as the Base (The "Advanced" Part)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False # Freeze the pre-trained weights

# 3. Build the Final Model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1), # MobileNetV2 expects pixels in [-1, 1]
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train
print("Training Advanced Transfer Learning Model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=15)

model.save('advanced_hand_model.h5')
