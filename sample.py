import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if TensorFlow can access GPU
print("GPU available:", tf.config.list_physical_devices('GPU'))
