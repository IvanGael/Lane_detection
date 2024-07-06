# reconstruct_model.py

import tensorflow as tf
from tensorflow import keras
import os

def create_model():
    model = keras.Sequential([
        keras.layers.BatchNormalization(
            input_shape=(80, 160, 3),
            name='batch_normalization_1'
        ),
        keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            name='Conv1'
        ),
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu',
            name='Conv2'
        )
    ])
    return model

# Create the model
model = create_model()

# Try to load weights
try:
    model.load_weights('model.h5')
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {str(e)}")

# Print model summary
model.summary()

# Save in .keras format
try:
    model.save('model.keras')
    print("Model saved successfully in .keras format.")
except Exception as e:
    print(f"Error saving model: {str(e)}")

# Optionally, export as SavedModel
try:
    tf.saved_model.save(model, 'saved_model')
    print("Model exported successfully as SavedModel.")
except Exception as e:
    print(f"Error exporting SavedModel: {str(e)}")

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
