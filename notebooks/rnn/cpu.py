
import sys
sys.path.append('.')
import tensorflow as tf
from functions import print_current_device, train_model, prepare_data, create_model
from metrics.metrics_class import MetricsCollector

epochs = 5

def main():
    trainX, trainY, testX, testY, scaler, dataset, look_back = prepare_data()
    model = create_model(look_back)
    metrics_collector = MetricsCollector(device='CPU', mac_model="M2", model="RNN")     # Setup metrics collector
    train_model(model, trainX, trainY, epochs, metrics_collector)
    metrics_collector.export_metrics()     # Export metrics

if __name__ == "__main__":
    tf.random.set_seed(7) # fix random seed for reproducibility
    tf.config.set_visible_devices([], 'GPU')  # Désactive le GPU
    print_current_device(tf)
    main()