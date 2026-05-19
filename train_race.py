import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. LOAD YOUR DATA (Mock Data for testing)
# ==========================================
# In reality, you will load your FairFace embeddings and labels here.
# X should be shape (num_photos, 512)
# y should be shape (num_photos,) containing numbers 0-6

import os

print("Loading FairFace data...")
if not os.path.exists("fairface_train_X.npy") or not os.path.exists("fairface_val_X.npy"):
    print("Error: Could not find the FairFace data files.")
    print("Please run 'python extract_fairface.py' first!")
    exit()

X_train = np.load("fairface_train_X.npy")
y_train = np.load("fairface_train_y.npy")

X_val = np.load("fairface_val_X.npy")
y_val = np.load("fairface_val_y.npy")

print(f"Successfully loaded {len(X_train)} training faces and {len(X_val)} validation faces!")  

# ==========================================
# 2. BUILD THE NEURAL NETWORK
# ==========================================
# We use a Multi-Layer Perceptron (MLP) because our input is just an array of numbers, not an image.

model = models.Sequential([
    # Input layer expects a 512-length array
    layers.Input(shape=(512,)),
    
    # Hidden Layer 1: 256 neurons to find patterns in the 512 numbers
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3), # Drops 30% of connections randomly to prevent overfitting
    
    # Hidden Layer 2: 128 neurons to refine the patterns
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    
    # Output Layer: 7 neurons (one for each race). Softmax converts outputs to percentages (0.0 to 1.0)
    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Used when labels are integers (0, 1, 2)
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 3. TRAIN THE MODEL WITH SMART CALLBACKS
# ==========================================
print("\nStarting training...")

# EarlyStopping: Stop training if the validation loss doesn't improve for 5 epochs
# ReduceLROnPlateau: If the validation loss stops improving, divide the learning rate by 2 to make finer adjustments
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, 
    y_train, 
    epochs=50, # We can safely set this higher because EarlyStopping will catch it!
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# ==========================================
# 4. SAVE THE MODEL
# ==========================================
model.save("race_classifier.keras")
print("\nModel successfully trained and saved as 'race_classifier.keras'!")
