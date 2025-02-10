import sys

sys.path.append(".")
import tensorflow as tf

from models.rnn.functions import create_model, prepare_data, train_model
from utils.config import MAC_MODEL, NUM_EPOCHS, print_current_device, select_gpu, NUM_EXPERIMENTS
from utils.metrics import MetricsCollector


def main():
    for _ in range(NUM_EXPERIMENTS):
        trainX, trainY, testX, testY, scaler, dataset, look_back = prepare_data()
        model = create_model(look_back)
        metrics_collector = MetricsCollector(
            device="GPU", mac_model=MAC_MODEL, model="RNN"
        )  # Setup metrics collector
        train_model(model, trainX, trainY, NUM_EPOCHS, metrics_collector)
        metrics_collector.export_metrics()
        metrics_collector.save_inference_time("rnn", model, testX)


if __name__ == "__main__":
    tf.random.set_seed(7)  # fix random seed for reproducibility
    select_gpu()
    print_current_device()
    main()