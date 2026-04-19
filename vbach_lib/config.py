import torch
from multiprocessing import cpu_count

class Config:
    """Конфигурация для VC"""
    
    def __init__(self, device_str: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        """
        Инициализация конфигурации
        
        Args:
            device_str: Строка устройства
        """
        self.device_str: str = device_str
        self.device_ids = None
        self.set_device(self.device_str)
        self.is_half: bool = False
        self.n_cpu: int = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def set_device(self, device_str: str) -> None:
        """
        Установить устройство
        
        Args:
            device_str: Строка устройства
        """
        if "cuda" in device_str.lower():
            if ":" in device_str:
                device_spec = device_str.split(":")[1]
                self.device_ids = [int(id) for id in device_spec.split(",") if id.isdigit()]
            else:
                self.device_ids = list(range(torch.cuda.device_count()))
            self.device = torch.device("cuda" if not self.device_ids else f"cuda:{self.device_ids[0]}")
        elif "mps" in device_str.lower():
            self.device_ids = None
            self.device = torch.device("mps")
        else:
            self.device_ids = None
            self.device = torch.device("cpu")

    def device_config(self):
        """
        Настройка параметров для устройства
        
        Returns:
            Кортеж (x_pad, x_query, x_center, x_max)
        """
        if self.device.type == "cuda":
            if self.device_ids:
                self.gpu_mem = self._configure_gpu(self.device_ids[0])

        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def _configure_gpu(self, device_id: int) -> int:
        """
        Настройка GPU
        
        Args:
            device_id: ID устройства
        
        Returns:
            Объем памяти GPU в GB
        """
        self.gpu_name = torch.cuda.get_device_name(f"cuda:{device_id}")
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        if (
            any(gpu in self.gpu_name for gpu in low_end_gpus)
            and "V100" not in self.gpu_name.upper()
        ):
            self.is_half = False
        return int(
            torch.cuda.get_device_properties(self.device).total_memory
            / 1024
            / 1024
            / 1024
            + 0.4
        )