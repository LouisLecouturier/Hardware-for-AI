import os
import subprocess
import time
import uuid
from datetime import datetime

import pandas as pd
import psutil


class MetricsCollector:
    def __init__(self, device, model, mac_model):
        self.model = model
        self.mac_model = mac_model
        self.device = device
        self.training_uuid = uuid.uuid4()
        self.metrics = {
            "timestamp": [],
            "memory_usage": [],
            "cpu_percent": [],
            "temperature": [],
            "training_time": [],
            "batch_time": [],
            "loss": [],
            "power_consumption": [],
            "epoch": [],
        }

    @staticmethod
    def get_temperature():
        try:
            # Utilisation de osx-cpu-temp via subprocess
            temp = subprocess.check_output(["osx-cpu-temp"]).decode().strip()
            temp = float(temp.replace("°C", ""))
            return temp
        except Exception:
            return None

    @staticmethod
    def get_power_consumption():
        try:
            # Utilisation de powermetrics via subprocess
            cmd = [
                "sudo",
                "powermetrics",
                "-n",
                "1",
                "--samplers",
                "cpu_power,gpu_power",
                "-i",
                "200",
            ]
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()

            # Extraction des valeurs de puissance
            cpu_power = 0
            gpu_power = 0

            for line in output.split("\n"):
                if "CPU Power" in line:
                    value = line.split(":")[1].strip()
                    if "mW" in value:
                        cpu_power = (
                            float(value.replace(" mW", "")) / 1000
                        )  # Conversion mW en W
                    else:
                        cpu_power = float(value.replace(" W", ""))
                elif "GPU Power" in line:
                    value = line.split(":")[1].strip()
                    if "mW" in value:
                        gpu_power = (
                            float(value.replace(" mW", "")) / 1000
                        )  # Conversion mW en W
                    else:
                        gpu_power = float(value.replace(" W", ""))

            total_power = cpu_power + gpu_power
            return total_power
        except Exception as e:
            print(e)
            return None

    def save_inference_time(self, task, model, x):
        """Gets the inference time of a model on a given input"""
        start_time = time.time()
        model.predict(x)
        inference_time = (time.time() - start_time) * 1000

        df = pd.DataFrame(
            {
                "mac_model": [self.mac_model],
                "device": [self.device],
                "inference_time_ms": [inference_time],
            }
        )

        file_path = f"exports/inference/inference_{task}_{self.device}.csv"

        # Vérifier si le fichier existe
        if os.path.exists(file_path):
            # Ajouter sans l'en-tête
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            # Créer nouveau fichier avec l'en-tête
            df.to_csv(file_path, index=False)

    def collect_system_metrics(self):
        self.metrics["timestamp"].append(datetime.now())
        self.metrics["memory_usage"].append(
            psutil.Process().memory_info().rss / 1024 / 1024
        )
        self.metrics["cpu_percent"].append(psutil.cpu_percent())
        self.metrics["temperature"].append(self.get_temperature())
        self.metrics["power_consumption"].append(self.get_power_consumption())

    def collect_training_metrics(self, batch_time, loss, epoch):
        self.metrics["batch_time"].append(batch_time)
        self.metrics["loss"].append(loss)
        self.metrics["epoch"].append(epoch)

    def export_metrics(self):
        metrics_to_export = {k: v for k, v in self.metrics.items() if len(v) > 0}
        df = pd.DataFrame(metrics_to_export)
        df["training_uuid"] = self.training_uuid

        file_path = f"exports/metrics_{self.model}_{self.device}_{self.mac_model}.csv"

        # Vérifier si le fichier existe
        if os.path.exists(file_path):
            # Ajouter sans l'en-tête
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            # Créer nouveau fichier avec l'en-tête
            df.to_csv(file_path, index=False)
            df.to_csv(file_path, index=False)
            df.to_csv(file_path, index=False)
            df.to_csv(file_path, index=False)
            df.to_csv(file_path, index=False)
