import os

import matplotlib.pyplot as plt
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv(".config")

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))
MAC_MODEL = os.getenv("MAC_MODEL", "M1")


def select_gpu():
    try:
        if len(tf.config.list_physical_devices("GPU")) > 0:
            # Utiliser with pour créer le contexte
            with tf.device("/device:GPU:0"):
                print("Utilisation du GPU (Metal)")
                return True
        else:
            with tf.device("/CPU:0"):
                print("GPU non disponible. Utilisation du CPU à la place.")
                return False
    except Exception:
        with tf.device("/CPU:0"):
            print("Erreur lors de l'activation du GPU. Utilisation du CPU à la place.")
            return False


def print_current_device():
    # Obtenir tous les devices disponibles
    devices = tf.config.list_physical_devices()
    print("Devices disponibles:", devices)

    # Vérifier le device actuel
    current_device = tf.config.get_visible_devices()
    print("Device actuel:", current_device)

    # Vérifier si un GPU est utilisé
    if tf.test.is_built_with_cuda():
        print("TensorFlow est configuré avec CUDA")

    # Vérifier les devices visibles (actifs)
    visible_gpus = tf.config.get_visible_devices("GPU")
    if len(visible_gpus) > 0:
        print("GPU/Metal est disponible et activé")
    else:
        print("CPU est utilisé")


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
    plt.show()
    plt.show()
    plt.show()
    plt.show()
