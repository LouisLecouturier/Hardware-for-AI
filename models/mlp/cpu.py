import sys

sys.path.append(".")

import tensorflow as tf

from models.mlp.functions import create_model, prepare_data, train_model
from utils.config import MAC_MODEL, NUM_EPOCHS, print_current_device
from utils.metrics import MetricsCollector


def main():
    trainX, trainY, testX, testY = prepare_data()
    model = create_model()
    metrics_collector = MetricsCollector(
        device="CPU", mac_model=MAC_MODEL, model="MLP"
    )  # Setup metrics collector
    train_model(model, trainX, trainY, NUM_EPOCHS, metrics_collector)
    metrics_collector.export_metrics()


if __name__ == "__main__":
    tf.random.set_seed(7)  # fix random seed for reproducibility
    tf.config.set_visible_devices([], "GPU")  # Désactive le GPU
    print_current_device()
    main()
