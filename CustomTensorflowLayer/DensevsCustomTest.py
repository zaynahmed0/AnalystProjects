from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import time
import os
import psutil  # For memory usage tracking
import matplotlib.pyplot as plt
import numpy as np
import struct
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
# Creating a directory to save the model results if it doesn't exist
results_dir = 'model(Conv3)_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load MNIST data
train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

X_train = read_idx(train_images_path)
y_train = read_idx(train_labels_path)
X_test = read_idx(test_images_path)
y_test = read_idx(test_labels_path)


# Normalize the input data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add a channel dimension to the data
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# Defining a custom TensorFlow layer with regularization
class CustomLayerRegularlizer(tf.keras.layers.Layer):

    def __init__(self, num_outputs, spread, regularization_factor=0.001, **kwargs):
        super(CustomLayerRegularlizer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.spread = spread
        self.regularization_factor = regularization_factor
        self.regularizer = tf.keras.regularizers.l2(self.regularization_factor)
        # Other initialization as needed

    def probability_curve(self, index, total_inputs):
        # Using a Gaussian distribution as an example
        x = tf.range(total_inputs, dtype=tf.float32)
        center = tf.constant(index, dtype=tf.float32)
        return tf.exp(-tf.square(x - center) / (2 * tf.square(self.spread)))


    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_inputs = tf.shape(inputs)[1]

        # Create a grid of indices for neurons and inputs
        neuron_indices = tf.range(self.num_outputs, dtype=tf.float32)
        input_indices = tf.range(num_inputs, dtype=tf.float32)
        neuron_grid, input_grid = tf.meshgrid(neuron_indices, input_indices, indexing='ij')

        # Compute the factors for all neurons at once
        factors = tf.exp(-tf.square(input_grid - neuron_grid) / (2 * tf.square(self.spread)))

        # Reshape inputs for batched matrix multiplication and apply factors
        expanded_inputs = tf.expand_dims(inputs, axis=1)  # Shape: [batch_size, 1, num_inputs]
        factors_expanded = tf.expand_dims(factors, axis=0)  # Shape: [1, num_outputs, num_inputs]
        weighted_inputs = expanded_inputs * factors_expanded

        # Sum across the inputs dimension to get the final output
        final_output = tf.reduce_sum(weighted_inputs, axis=2)  # Shape: [batch_size, num_outputs]

        #Computer regularizer loss
        reg_loss = self.regularizer(self.weights)  # Assuming 'self.weights' contains the layer's weights
        self.add_loss(reg_loss)

        return final_output

# Function to add Gaussian noise
def add_gaussian_noise(X, noise_factor=0.1):
    noisy_data = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    return np.clip(noisy_data, 0, 1)
# Define a function to create a model
def create_cnn_model(use_custom_layer, num_custom_layers, num_filters, regularization_factor, learning_rate):
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28 with 1 channel
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)

    # Encoder
    x = Dense(num_filters, activation='relu')(x)

    for _ in range(num_custom_layers):
        if use_custom_layer:
            x = CustomLayerRegularlizer(num_outputs=num_filters, spread=1.0,
                                       regularization_factor=regularization_factor)(x)
        else:
            x = Dense(num_filters, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(regularization_factor))(x)

    x = Dense(128, activation='relu')(x)

    # Decoder
    # Assuming the decoder mirrors the encoder structure


    # Output layer: Reconstructing the input
    output_layer = Dense(10, activation='softmax')(x)

    # Create the autoencoder model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


use_custom_layer_options = [True, False]
num_custom_layers_options = [1,5,10]  # Adjust based on whether your custom layer is designed for CNN
num_filters_options = [32, 64]  # Using number of filters instead of units for convolutional layers
regularization_factors = [0.001, 0.01, 0.1]

epoch_options = [10, 20]  # Keeping epochs as before
batch_size_options = [64, 128]  # Keeping batch sizes as before
learning_rate_options = [0.001, 0.01]  # Keeping learning rates as before

# Adjusted total trials calculation
total_trials = len(use_custom_layer_options) * len(num_custom_layers_options) * len(num_filters_options) * len(regularization_factors) * len(epoch_options) * len(batch_size_options) * len(learning_rate_options)
Trial = 1
# Store results and history objects
results = []
history_objects = {}  # Dictionary to store history objects
skipped_trials = []


# Iterate over all configurations
for num_custom_layers in num_custom_layers_options:
    for num_filters in num_filters_options:
        for reg_factor in regularization_factors:
            for epochs in epoch_options:
                for batch_size in batch_size_options:
                    for learning_rate in learning_rate_options:
                        history_data = []
                        for use_custom_layer in use_custom_layer_options:
                            try:
                                print(f"\nTrial {Trial}/{total_trials}: Training with configuration: Use Custom Layer={use_custom_layer}, Num Custom Layers={num_custom_layers}, Num Filters={num_filters}, Regularization Factor={reg_factor}, Epochs={epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}")
                                Trial += 1

                                model = create_cnn_model(use_custom_layer, num_custom_layers, num_filters, reg_factor, learning_rate)

                                process = psutil.Process(os.getpid())
                                mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
                                start_time = time.time()  # Start time

                                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

                                training_time = time.time() - start_time  # Training time
                                mem_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
                                peak_memory_usage = mem_after - mem_before

                                loss, accuracy = model.evaluate(X_test, y_test)

                                model_complexity = model.count_params()  # Model complexity

                                # Append results
                                results.append({
                                    'Use Custom Layer': use_custom_layer,
                                    'Num Custom Layers': num_custom_layers,
                                    'Num Filters': num_filters,
                                    'Regularization Factor': reg_factor,
                                    'Epochs': epochs,
                                    'Batch Size': batch_size,
                                    'Learning Rate': learning_rate,
                                    'Accuracy': accuracy,
                                    'Loss': loss,
                                    'Training Time': training_time,
                                    'Peak Memory Usage (MB)': peak_memory_usage,
                                    'Model Complexity': model_complexity
                                })

                                key = (use_custom_layer, num_custom_layers, num_filters, reg_factor, epochs, batch_size, learning_rate)
                                history_objects[key] = history.history
                                history_data.append((use_custom_layer, history.history))
                            except Exception as e:
                                print(f"Error encountered in trial {Trial}: {e}. Skipping this trial.")
                                skipped_trials.append({
                                    'Trial': Trial,
                                    'Use Custom Layer': use_custom_layer,
                                    'Num Custom Layers': num_custom_layers,
                                    'Num Filters': num_filters,
                                    'Regularization Factor': reg_factor,
                                    'Epochs': epochs,
                                    'Batch Size': batch_size,
                                    'Learning Rate': learning_rate,
                                    'Error Message': str(e)
                                })

                                continue  # Skip to the next iteration
                        if len(history_data) == 2:  # Ensuring we have both histories
                            # Unpack histories for custom and dense layers
                            custom_layer_history, dense_layer_history = history_data[0][1], history_data[1][1]

                            # Create figure for accuracy plot
                            plt.figure(figsize=(12, 6))

                            # Accuracy subplot
                            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
                            plt.plot(custom_layer_history['accuracy'], label='Custom Layer Train Accuracy')
                            plt.plot(custom_layer_history['val_accuracy'], label='Custom Layer Val Accuracy')
                            plt.plot(dense_layer_history['accuracy'], label='Dense Layer Train Accuracy',
                                     linestyle='--')
                            plt.plot(dense_layer_history['val_accuracy'], label='Dense Layer Val Accuracy',
                                     linestyle='--')
                            plt.title(
                                f"Accuracy (Custom vs Dense)\nCL: {num_custom_layers}, NF: {num_filters}, RF: {reg_factor}, EP: {epochs}, BS: {batch_size}, LR: {learning_rate}")
                            plt.xlabel('Epoch')
                            plt.ylabel('Accuracy')
                            plt.legend(loc='lower right')

                            # Loss subplot
                            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
                            plt.plot(custom_layer_history['loss'], label='Custom Layer Train Loss')
                            plt.plot(custom_layer_history['val_loss'], label='Custom Layer Val Loss')
                            plt.plot(dense_layer_history['loss'], label='Dense Layer Train Loss', linestyle='--')
                            plt.plot(dense_layer_history['val_loss'], label='Dense Layer Val Loss', linestyle='--')
                            plt.title(
                                f"Loss (Custom vs Dense)\nCL: {num_custom_layers}, NF: {num_filters}, RF: {reg_factor}, EP: {epochs}, BS: {batch_size}, LR: {learning_rate}")
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.legend(loc='upper right')

                            plt.tight_layout()

                            # Save plot to the results directory
                            plot_filename = f"learning_curve_{num_custom_layers}_{num_filters}_{reg_factor}_{epochs}_{batch_size}_{learning_rate}.png"
                            plt.savefig(os.path.join(results_dir, plot_filename))
                            plt.close()
if skipped_trials:
    df_skipped_trials = pd.DataFrame(skipped_trials)
    skipped_csv_filename = os.path.join(results_dir, 'skipped(Conv2)_trials.csv')
    df_skipped_trials.to_csv(skipped_csv_filename, index=False)
    print(f"Skipped trials saved to {skipped_csv_filename}.")


# Converting the collected results into a DataFrame and saving it
df_results = pd.DataFrame(results)
csv_filename = os.path.join(results_dir, '../model(Conv3)_comparison_results.csv')
df_results.to_csv(csv_filename, index=False)
