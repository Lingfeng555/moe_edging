import time
from pyJoules.energy_meter import EnergyMeter
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
import psutil
import GPUtil


class Metricas:
    def __init__(self):

        self.meter = None
        self.start_time = None
        self.domains = [RaplPackageDomain(0), RaplDramDomain(0)]

        try:
            self.domains.append(NvidiaGPUDomain(0))

        except:
            print("No se detectó GPU NVIDIA")

    def get_cpu_metrics(self):

        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)

        return f"CPU Uso: {cpu_percent}%, Frecuencia: {cpu_freq.current:.2f} MHz"

    def get_gpu_metrics(self):

        gpus = GPUtil.getGPUs()

        if gpus:
            gpu = gpus[0] 

            return f"GPU Uso: {gpu.load*100:.2f}%, Memoria: {gpu.memoryUsed}/{gpu.memoryTotal} MB"

        return "No se detectó GPU NVIDIA"

    def iniciar_monitoreo(self):

        self.meter = EnergyMeter(self.domains)
        self.meter.start()
        self.start_time = time.time()

    def obtener_metricas(self):

        current_time = time.time()
        duration = current_time - self.start_time
        
        cpu_metrics = self.get_cpu_metrics()
        gpu_metrics = self.get_gpu_metrics()
        
        energy_trace = self.meter.get_trace()
        total_energy = sum(sample.energy['package_0'] for sample in energy_trace)
        
        power = total_energy / (duration * 1e6)  # Convertir μJ a W
        
        return {
            "duration": duration,
            "total_energy": total_energy / 1e6,  # Convertir a J
            "power": power,
            "cpu_metrics": cpu_metrics,
            "gpu_metrics": gpu_metrics
        }

    def detener_monitoreo(self):

        self.meter.stop()
        trace = self.meter.get_trace()

        return trace

def monitor_system(self):

    print("Iniciando monitoreo. Presiona Ctrl+C para detener.")
    seguir = True
    self.iniciar_monitoreo()

    try:
        while seguir:
            time.sleep(1)  # Actualizar cada segundo
            metricas = self.obtener_metricas()
            
            print(f"\nTiempo transcurrido: {metricas['duration']:.2f} s")
            print(f"Consumo de energía: {metricas['total_energy']:.6f} J")
            print(f"Potencia promedio: {metricas['power']:.2f} W")
            print(metricas['cpu_metrics'])
            print(metricas['gpu_metrics'])
    
    except KeyboardInterrupt:
        print("\nDetención del monitoreo...")

    finally:
        seguir = False
        trace = self.detener_monitoreo()
        
        for sample in trace:
            print(f"Muestra: {sample}")


if __name__ == "__main__":
    metricas = Metricas()
    metricas.monitor_system()
