import sys
sys.path.append('.')
import tensorflow as tf
from functions import print_current_device, train_model, prepare_data, create_model
from metrics.metrics_class import MetricsCollector

epochs = 5


def main():
    trainX, trainY, testX, testY, scaler, dataset, look_back = prepare_data()
    model = create_model(look_back)
    metrics_collector = MetricsCollector(device='GPU', mac_model="M2", model="RNN")     # Setup metrics collector
    train_model(model, trainX, trainY, epochs, metrics_collector)
    metrics_collector.export_metrics()  
  

def select_gpu():
    try:
        if len(tf.config.list_physical_devices('GPU')) > 0:
            # Utiliser with pour créer le contexte
            with tf.device('/device:GPU:0'):
                print("Utilisation du GPU (Metal)")
                return True
        else:
            with tf.device('/CPU:0'):
                print("GPU non disponible. Utilisation du CPU à la place.")
                return False
    except:
        with tf.device('/CPU:0'):
            print("Erreur lors de l'activation du GPU. Utilisation du CPU à la place.")
            return False



if __name__ == "__main__":
    tf.random.set_seed(7) # fix random seed for reproducibility
    select_gpu()
    print_current_device(tf)
    main()