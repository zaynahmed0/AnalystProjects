
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import keras_tuner as kt

# Load Data
file_path = '../mitbih_train.csv'  # Replace with your file path
dataframe = pd.read_csv(file_path, header=None)  # Assuming no header
data = dataframe.values

# Assume the last column is the label and remove it
X = data[:, :-1]

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Define the Autoencoder architecture as a function
def build_model(hp):
    input_dim = X_train.shape[1]  # Number of features
    input_layer = Input(shape=(input_dim,))

    # First Dense Encoder Layer
    encoder = Dense(units=hp.Int('units_encoder', min_value=32, max_value=256, step=32),
                    activation='relu')(input_layer)

    # Additional Dense Layer

    # Bottleneck Layer
    bottleneck = Dense(units=hp.Int('units_bottleneck', min_value=4, max_value=32, step=4),
                       activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(encoder)

    # Dense Decoder Layer
    decoder = Dense(units=hp.Int('units_decoder', min_value=32, max_value=256, step=32),
                    activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(bottleneck)
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return autoencoder


tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=30,
                     factor=3,
                     directory='my_dir',
                     project_name='autoencoder')

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Execute the hyperparameter search
tuner.search(X_train, X_train,
             epochs=50,
             batch_size=256,
             shuffle=True,
             validation_data=(X_test, X_test),
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the encoder is {best_hps.get('units_encoder')},
in the bottleneck is {best_hps.get('units_bottleneck')}, and in the decoder is {best_hps.get('units_decoder')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, X_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))



# Evaluate the model
reconstructions = model.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
print("Reconstruction MSE:", np.mean(mse))

# Plotting Original ECG and Reconstructed ECG
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.plot(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.plot(reconstructions[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

