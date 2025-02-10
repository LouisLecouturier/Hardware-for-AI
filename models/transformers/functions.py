import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D



def train_model(model, trainX, trainY, epochs, metrics_collector):
    for epoch in range(epochs):
        start_time = time.time()
        metrics_collector.collect_system_metrics()

        history = model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

        batch_time = time.time() - start_time

        print(f"Batch time: {batch_time}")
        metrics_collector.collect_training_metrics(
            batch_time=batch_time,
            loss=history.history["loss"][-1],
            epoch=epoch
        )


def prepare_data():
    # load the dataset
    path = os.path.join("datasets", "airline-passengers.csv")
    dataframe = pd.read_csv(path, usecols=[1], engine="python")
    dataset = dataframe.values
    dataset = dataset.astype("float32")

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]
    print(len(train), len(test))

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i : (i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # reshape into X=t and Y=t+1
    look_back = 10
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY, scaler, dataset, look_back

def create_transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention multi-têtes
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size
    )(inputs, inputs)
    
    # Add & Normalize
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward
    outputs = Dense(ff_dim, activation="relu")(attention_output)
    outputs = Dense(inputs.shape[-1])(outputs)
    
    # Add & Normalize
    outputs = LayerNormalization(epsilon=1e-6)(attention_output + outputs)
    return outputs



def create_model(look_back, tf):
    # Paramètres du Transformer
    head_size = 256
    num_heads = 4
    ff_dim = 4
    
    # Input
    inputs = Input(shape=(1, look_back))
    
    # Transformer block
    transformer_block = create_transformer_block(
        inputs, head_size, num_heads, ff_dim
    )
    
    # Global pooling
    x = GlobalAveragePooling1D()(transformer_block)
    
    # Output
    outputs = Dense(1)(x)
    
    # Création du modèle
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )
    
    return model


def plot_comparison(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    devices = list(results.keys())

    # Temps d'entraînement
    times = [results[d]["training_time"] for d in devices]
    ax1.bar(devices, times)
    ax1.set_title("Temps total d'entraînement")
    ax1.set_ylabel("Secondes")

    # Utilisation mémoire
    memory = [results[d]["memory_used"] for d in devices]
    ax2.bar(devices, memory)
    ax2.set_title("Utilisation mémoire")
    ax2.set_ylabel("MB")

    # Courbe de loss
    for device in devices:
        ax3.plot(results[device]["loss_history"], label=device)
    ax3.set_title("Convergence de la loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()

    # CPU Usage
    cpu_usage = [results[d]["cpu_usage"] for d in devices]
    ax4.bar(devices, cpu_usage)
    ax4.set_title("Utilisation CPU moyenne")
    ax4.set_ylabel("%")

    plt.tight_layout()
    plt.show()


# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

# # calculate root mean squared error
# trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))


# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset), label='Dataset')
# plt.plot(trainPredictPlot, label='Train')
# plt.plot(testPredictPlot, label='Test')
# plt.legend()
# plt.show()
