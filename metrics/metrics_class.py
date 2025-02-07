import psutil
import time
import numpy as np
from datetime import datetime
import pandas as pd
import subprocess


def get_temperature():
    try:
        # Utilisation de osx-cpu-temp via subprocess
        temp = subprocess.check_output(['osx-cpu-temp']).decode().strip()
        temp = float(temp.replace('°C', ''))
        return temp
    except:
        return None

def get_power_consumption():
    try:
        # Utilisation de powermetrics via subprocess
        cmd = ["sudo", "powermetrics", "-n", "1", "--samplers", "cpu_power,gpu_power", "-i", "200"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        
        # Extraction des valeurs de puissance
        cpu_power = 0
        gpu_power = 0
        
        for line in output.split('\n'):
            if 'CPU Power' in line:
                value = line.split(':')[1].strip()
                if 'mW' in value:
                    cpu_power = float(value.replace(' mW', '')) / 1000  # Conversion mW en W
                else:
                    cpu_power = float(value.replace(' W', ''))
            elif 'GPU Power' in line:
                value = line.split(':')[1].strip()
                if 'mW' in value:
                    gpu_power = float(value.replace(' mW', '')) / 1000  # Conversion mW en W
                else:
                    gpu_power = float(value.replace(' W', ''))
        
        total_power = cpu_power + gpu_power
        return total_power
    except Exception as e:
        print(e)
        return None


class MetricsCollector:
    def __init__(self, device, model, mac_model):
        self.model = model
        self.mac_model = mac_model
        self.device = device
        self.metrics = {
            'timestamp': [],
            'memory_usage': [],
            'cpu_percent': [],
            'temperature': [],
            'training_time': [],
            'batch_time': [],
            'loss': [],
            'power_consumption': []
        }

    def collect_system_metrics(self):
        self.metrics['timestamp'].append(datetime.now())
        self.metrics['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['temperature'].append(get_temperature())
        self.metrics['power_consumption'].append(get_power_consumption())

    def collect_training_metrics(self, batch_time, loss):
        self.metrics['batch_time'].append(batch_time)
        self.metrics['loss'].append(loss)

    def export_metrics(self):
        metrics_to_export = {k: v for k, v in self.metrics.items() if len(v) > 0}
        df = pd.DataFrame(metrics_to_export)
        print(df)
        df.to_csv(f'exports/metrics_{self.model}_{self.device}_{self.mac_model}.csv', mode="a", index=False)