import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# --- YOUR EVOLVED OPTIMAL VALUES ---
EVOLVED_LR = 0.01  
EVOLVED_FILTERS = 16

# 1. Load Data
img_size = (64, 64)
print("Loading dataset for final training...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset', validation_split=0.2, subset="training", seed=123,
    image_size=img_size, batch_size=32, label_mode='binary'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset', validation_split=0.2, subset="validation", seed=123,
    image_size=img_size, batch_size=32, label_mode='binary'
)

# 2. Final Deep Architecture
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Rescaling(1./255),
    
    # Block 1: Using your evolved 16 filters
    layers.Conv2D(EVOLVED_FILTERS, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2: Adding depth for robustness
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5), # Crucial for preventing overfitting
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=EVOLVED_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. Full Training (15 Epochs)
print(f"Final training started with evolved LR: {EVOLVED_LR}")
history = model.fit(train_ds, validation_data=val_ds, epochs=15)

# 4. Save and Generate Graph for Report
model.save('final_hand_model.h5')
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Final Model Performance (Evolved Architecture)')
plt.legend()
plt.savefig('final_training_graph.png')
print(" Success! final_hand_model.h5 is ready.")
