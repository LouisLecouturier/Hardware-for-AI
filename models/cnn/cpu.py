import sys

sys.path.append(".")

import tensorflow as tf

from models.cnn.functions import create_model, prepare_data, train_model
from utils.config import MAC_MODEL, NUM_EPOCHS, NUM_EXPERIMENTS, print_current_device
from utils.metrics import MetricsCollector


def main():
    for _ in range(NUM_EXPERIMENTS):
        trainX, trainY, testX, testY = prepare_data()
        model = create_model()
        metrics_collector = MetricsCollector(
            device="CPU", mac_model=MAC_MODEL, model="CNN"
        )  # Setup metrics collector
        train_model(model, trainX, trainY, NUM_EPOCHS, metrics_collector)
        metrics_collector.export_metrics()
        metrics_collector.save_inference_time("cnn", model, testX)


if __name__ == "__main__":
    tf.random.set_seed(7)  # fix random seed for reproducibility
    tf.config.set_visible_devices([], "GPU")  # Désactive le GPU
    print_current_device()
    main()
    main()
    main()
    main()
    main()
    main()
    main()
    main()
    main()
