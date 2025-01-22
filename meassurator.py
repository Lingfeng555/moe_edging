import sys
import time
import subprocess
import pandas as pd
import numpy as np
import psutil
import os
from datetime import datetime


class CPUWatcher :
    def __init__(self, max_rows:int = 10000):
        self.max_rows = max_rows
        self.general_data = {"CPU_usage": [], 
                             "Memory_info": [],
                             "DiskUsage": [],
                             "nWattsSupply": [],
                             "Moment": [],
                            }
        self.n_saves = 0

        if not os.path.exists(f'outputs/cpu'):
            os.makedirs(f'outputs/cpu')
            os.makedirs(f'outputs/cpu/threads')
            os.makedirs(f'outputs/cpu/cores')
            os.makedirs(f'outputs/cpu/general')
        
    def get_cpu_core_data(self):
        """
        Retorna dos DataFrames:
        - df_fisico: datos agregados por núcleo físico, con promedio de uso, frecuencia y temperatura.
        - df_logico: datos individuales por núcleo lógico (hilo).
        Se asume que los núcleos lógicos están ordenados de forma consecutiva y se agrupan equitativamente.
        """
        # Número total de núcleos físicos y lógicos
        num_cores_fisicos = psutil.cpu_count(logical=False)
        num_cores_logicos = psutil.cpu_count(logical=True)
        
        # Uso de cada núcleo lógico (hilo)
        cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
        
        # Frecuencia actual de cada núcleo lógico
        cpu_frequencies = psutil.cpu_freq(percpu=True)
        
        # Temperaturas (si están disponibles)
        try:
            temps_dict = psutil.sensors_temperatures()
            temps = temps_dict.get("coretemp", [])
            # Se extraen las temperaturas en orden, si se cuentan con tantos sensores como núcleos lógicos
            temps_values = [sensor.current for sensor in temps]
        except AttributeError:
            temps_values = []
        
        # Crear DataFrame para núcleos lógicos
        data_logico = []
        for i in range(num_cores_logicos):
            freq = cpu_frequencies[i].current if cpu_frequencies and i < len(cpu_frequencies) else None
            temp = temps_values[i] if temps_values and i < len(temps_values) else None
            data_logico.append({
                "Núcleo lógico": i,
                "Uso (%)": cpu_percentages[i],
                "Frecuencia (MHz)": freq,
                "Temperatura (°C)": temp,
            })
        df_logico = pd.DataFrame(data_logico)

        grupos = np.array_split(df_logico, num_cores_fisicos)
        
        data_fisico = []
        for idx, grupo in enumerate(grupos):
            uso_prom = grupo["Uso (%)"].mean()
            freq_prom = grupo["Frecuencia (MHz)"].mean() if grupo["Frecuencia (MHz)"].notna().any() else None
            temp_prom = grupo["Temperatura (°C)"].mean() if grupo["Temperatura (°C)"].notna().any() else None
            data_fisico.append({
                "Núcleo físico": idx,
                "Uso (%)": uso_prom,
                "Frecuencia (MHz)": freq_prom,
                "Temperatura (°C)": temp_prom,
                "Hilos asociados": len(grupo)
            })
        df_fisico = pd.DataFrame(data_fisico)
        
        return df_fisico, df_logico

    def read_power_supply(self):
        path = "/sys/class/power_supply/BAT0/"
        try:
            with open(path + "power_now", "r") as power_file:
                power_now = float(power_file.read().strip())  # En microwatios
            return (power_now / 1e6)
        except FileNotFoundError:
            return -1
        
    def record_general_data(self):
        self.general_data["CPU_usage"].append(psutil.cpu_percent(interval=1))
        self.general_data["Memory_info"].append(psutil.virtual_memory())
        self.general_data["DiskUsage"].append(psutil.disk_usage('/'))
        self.general_data["nWattsSupply"].append(self.read_power_supply())
        self.general_data["Moment"].append(str(datetime.now()))
    
    def record_cpu_data(self):
        df_threads, df_cores = self.get_cpu_core_data()
        df_threads.to_csv(f'outputs/cpu/threads/{datetime.now()}.csv', index=False)
        df_cores.to_csv(f'outputs/cpu/cores/{datetime.now()}.csv', index=False)

    def record(self):
        self.record_general_data()
        self.record_cpu_data()
        if(len(self.general_data["CPU_usage"])>self.max_rows): self.save()

    def save(self):
        pd.DataFrame(self.general_data).to_csv(f'outputs/cpu/general/general_part_{self.n_saves}.csv', index=False)
        self.n_saves += 1

class GPUWatcher:
    def __init__(self, max_rows:int = 10000, pids : list = []):
        if not os.path.exists('outputs/gpu'):
            os.makedirs('outputs/gpu')
        self.max_rows = max_rows
        self.n_saves = 0
        self.headers = [
            "timestamp",
            "name",
            "temperature.gpu",
            "utilization.gpu",
            "utilization.memory",
            "memory.total",
            "memory.used",
            "memory.free",
            "power.draw",
            "fan.speed",
            "pstate",
            "clocks.current.graphics",
            "clocks.current.memory"
        ]
        self.general_data = []
        self.general_command = [
            "nvidia-smi",
            "--query-gpu=" + ",".join(self.headers),
            "--format=csv,noheader,nounits"
        ]
        self.pid_command = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"
        ]
        self.watching_pids = {}
        print(pids)
        if len(pids) > 0 : 
            for pid in pids: 
                self.watching_pids[str(pid)] = []
                if not os.path.exists(f'outputs/gpu/{pid}'):
                    os.makedirs(f'outputs/gpu/{pid}')
    def record(self):
        # Ejecuta el comando y captura la salida en texto
        resultado = subprocess.run(self.general_command, capture_output=True, text=True)

        # Separa en líneas y cada línea en columnas
        lineas = resultado.stdout.strip().split("\n")
        for linea in lineas:
            fila = [valor.strip() for valor in linea.split(",")]
            self.general_data.append(fila)
        self.meassure_pids()
        
        if len(self.general_data) > self.max_rows: self.save()
    
    def save(self):
        pd.DataFrame(self.general_data, columns=self.headers).to_csv(f"outputs/gpu/gpu_metrics_part_{self.n_saves}.csv", index=False)

        if(len(self.watching_pids) > 0):
            for pid in self.watching_pids:
                pd.DataFrame(self.watching_pids[pid], columns=["pid", "process_name", "used_gpu_memory"]).to_csv(f"outputs/gpu/{pid}/gpu_pid_metrics_part_{self.n_saves}.csv", index=False)
                self.watching_pids[pid].clear()
        self.n_saves +=1
        self.general_data.clear()

    def meassure_pids(self):
        resultado = subprocess.run(self.pid_command, capture_output=True, text=True)
        lineas = resultado.stdout.strip().split("\n")
        for linea in lineas:
            columnas = [col.strip() for col in linea.split(",")]
            if columnas[0] in self.watching_pids.keys():
                self.watching_pids[columnas[0]].append(columnas)

class Watcher:
    @staticmethod
    def measure_energy_consumption(executable_file_path: str, timelapse: int = 1):
        """Ejecuta un script y mide consumo de CPU/GPU cada cierto tiempo."""
        proc = subprocess.Popen(['python3', executable_file_path])
        gpu_watcher = GPUWatcher(pids=[proc.pid])
        cpu_watcher = CPUWatcher()
        while proc.poll() is None:
            # Procesar o almacenar gpu_data y cpu_data
            gpu_watcher.record()
            cpu_watcher.record()
            time.sleep(timelapse)
        gpu_watcher.save()
        cpu_watcher.save()
        return proc.returncode

if __name__ == "__main__":
    print("Uso: python3 meassurator.py <path_del_target_a_medir> [timelapse]")
    if len(sys.argv) < 2:
        print("Uso: python3 meassurator.py <path_del_target_a_medir> [timelapse]")
        sys.exit(1)
    script_path = sys.argv[1]
    timelapse = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    Watcher.measure_energy_consumption(script_path, timelapse)
