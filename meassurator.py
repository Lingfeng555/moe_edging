import sys
import time
import subprocess
import pandas as pd
import os

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
        watcher = GPUWatcher(pids=[proc.pid])
        while proc.poll() is None:
            # Procesar o almacenar gpu_data y cpu_data
            watcher.record()
            time.sleep(timelapse)
        watcher.save()
        return proc.returncode

if __name__ == "__main__":
    print("Uso: python3 meassurator.py <path_del_target_a_medir> [timelapse]")
    if len(sys.argv) < 2:
        print("Uso: python3 meassurator.py <path_del_target_a_medir> [timelapse]")
        sys.exit(1)
    script_path = sys.argv[1]
    timelapse = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    Watcher.measure_energy_consumption(script_path, timelapse)
