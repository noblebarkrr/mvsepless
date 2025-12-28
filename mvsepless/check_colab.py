#@title Проверка среды выполнения
import sys
import os
import json
import warnings
import platform
import socket
import uuid
import pkgutil
import importlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ColabDetectionMethod(Enum):
    """Методы детектирования Colab"""
    ENV_VARIABLES = "env_variables"
    PYTHON_MODULES = "python_modules"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    RUNTIME_INFO = "runtime_info"
    GPU_INFO = "gpu_info"
    CUSTOM_CHECKS = "custom_checks"


@dataclass
class ColabDetectionResult:
    """Результат детектирования Colab"""
    is_colab: bool
    confidence: float  # 0.0 - 1.0
    detection_methods: Dict[ColabDetectionMethod, bool]
    colab_version: Optional[str] = None
    runtime_type: Optional[str] = None  # 'GPU', 'TPU', 'CPU'
    colab_details: Dict[str, Any] = None
    warnings: list = None
    
    def __post_init__(self):
        if self.colab_details is None:
            self.colab_details = {}
        if self.warnings is None:
            self.warnings = []


class ColabDetector:
    """Продвинутый детектор Google Colab среды"""
    
    # Сигнатуры Colab
    COLAB_ENV_VARS = {
        'COLAB_RELEASE_TAG': None,
        'COLAB_PYTHON_VERSION': None,
        'COLAB_GPU': None,
        'COLAB_TPU': None,
    }
    
    COLAB_PATHS = [
        '/content',
        '/content/drive',
        '/content/drive/MyDrive',
        '/usr/local/lib/python3.*/dist-packages/google/colab'
    ]
    
    COLAB_MODULES = [
        'google.colab',
        'google.colab._system_commands',
        'google.colab._shell',
        'google.colab.output',
        'google.colab.widgets',
    ]
    
    COLAB_NETWORK_DOMAINS = [
        'colab.research.google.com',
        'colab.googleusercontent.com',
    ]
    
    # Коллекция IP диапазонов Google
    GOOGLE_IP_RANGES = [
        '8.8.8.8/16',  # Google DNS
        '172.217.0.0/16',
        '74.125.0.0/16',
    ]
    
    def __init__(self, enable_network_checks: bool = True, 
                 enable_gpu_check: bool = True,
                 enable_tpu_check: bool = True):
        self.enable_network_checks = enable_network_checks
        self.enable_gpu_check = enable_gpu_check
        self.enable_tpu_check = enable_tpu_check
        self._cache = {}
        self._results = {}
        
    def _check_env_variables(self) -> Tuple[bool, float, Dict]:
        """Проверка переменных окружения"""
        detected_vars = {}
        confidence = 0.0
        
        for var in self.COLAB_ENV_VARS:
            if var in os.environ:
                detected_vars[var] = os.environ[var]
                confidence += 0.15
        
        # Специфичные для Colab переменные
        colab_specific = [
            'GCS_READ_CACHE_BLOCK_SIZE_MB',
            'GCS_CACHING_ENDPOINT',
            'LD_PRELOAD'
        ]
        
        for var in colab_specific:
            if var in os.environ:
                detected_vars[var] = "Present"
                confidence += 0.05
        
        is_colab = len(detected_vars) > 0
        confidence = min(confidence, 1.0)
        
        return is_colab, confidence, detected_vars
    
    def _check_python_modules(self) -> Tuple[bool, float, Dict]:
        """Проверка Python модулей"""
        detected_modules = {}
        confidence = 0.0
        
        for module_name in self.COLAB_MODULES:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    detected_modules[module_name] = True
                    
                    # Попытка получить версию
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, '__version__'):
                            detected_modules[f'{module_name}_version'] = module.__version__
                    except:
                        pass
                    
                    confidence += 0.3
            except:
                continue
        
        # Проверка специфичных пакетов Colab
        colab_packages = ['google-colab', 'colab-code']
        for pkg in colab_packages:
            try:
                dist = pkgutil.get_distribution(pkg)
                if dist:
                    detected_modules[pkg] = dist.version
                    confidence += 0.2
            except:
                pass
        
        is_colab = len(detected_modules) > 0
        confidence = min(confidence, 1.0)
        
        return is_colab, confidence, detected_modules
    
    def _check_filesystem(self) -> Tuple[bool, float, Dict]:
        """Проверка файловой системы"""
        detected_paths = {}
        confidence = 0.0
        
        # Проверка путей
        for path_pattern in self.COLAB_PATHS:
            if '*' in path_pattern:
                import glob
                matches = glob.glob(path_pattern)
                if matches:
                    detected_paths[path_pattern] = matches
                    confidence += 0.2
            elif os.path.exists(path_pattern):
                detected_paths[path_pattern] = True
                confidence += 0.3
        
        # Проверка специфичных файлов
        colab_files = [
            '/etc/colab_release',
            '/etc/colab-version',
            '/usr/local/bin/colab'
        ]
        
        for file_path in colab_files:
            if os.path.exists(file_path):
                detected_paths[file_path] = True
                confidence += 0.4
                
                # Чтение файла с версией
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        detected_paths[f'{file_path}_content'] = content[:100]
                except:
                    pass
        
        
        is_colab = len(detected_paths) > 0
        confidence = min(confidence, 1.0)
        
        return is_colab, confidence, detected_paths
    
    def _check_network(self) -> Tuple[bool, float, Dict]:
        """Проверка сетевых признаков"""
        if not self.enable_network_checks:
            return False, 0.0, {'network_checks_disabled': True}
        
        detected_network = {}
        confidence = 0.0
        
        # Проверка hostname
        try:
            hostname = socket.gethostname()
            detected_network['hostname'] = hostname
            
            if 'colab' in hostname.lower():
                confidence += 0.4
        except:
            pass
        
        # Проверка DNS
        try:
            import dns.resolver
            for domain in self.COLAB_NETWORK_DOMAINS:
                try:
                    answers = dns.resolver.resolve(domain, 'A')
                    if answers:
                        detected_network[f'dns_{domain}'] = [str(r) for r in answers]
                        confidence += 0.2
                except:
                    pass
        except ImportError:
            detected_network['dns_check'] = 'dnspython not installed'
        
        # Проверка доступности Colab доменов через HTTP
        try:
            import urllib.request
            import ssl
            
            # Создаем кастомный контекст для избежания SSL ошибок
            context = ssl._create_unverified_context()
            
            test_urls = [
                'https://colab.research.google.com/',
                'https://clients6.google.com/'
            ]
            
            for url in test_urls:
                try:
                    req = urllib.request.Request(
                        url,
                        headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
                    )
                    with urllib.request.urlopen(req, context=context, timeout=5) as response:
                        if response.status == 200:
                            detected_network[f'http_{url}'] = response.status
                            confidence += 0.15
                except:
                    continue
        except ImportError:
            pass
        
        # Проверка метаданных Google Cloud
        try:
            import requests
            metadata_url = "http://metadata.google.internal/computeMetadata/v1/"
            headers = {"Metadata-Flavor": "Google"}
            
            try:
                response = requests.get(
                    metadata_url,
                    headers=headers,
                    timeout=3
                )
                if response.status_code == 200:
                    detected_network['gcp_metadata'] = True
                    confidence += 0.35
                    
                    # Получение дополнительных метаданных
                    meta_endpoints = [
                        "instance/attributes/colab-version",
                        "project/project-id"
                    ]
                    
                    for endpoint in meta_endpoints:
                        try:
                            resp = requests.get(
                                f"{metadata_url}{endpoint}",
                                headers=headers,
                                timeout=2
                            )
                            if resp.status_code == 200:
                                detected_network[f'metadata_{endpoint}'] = resp.text.strip()
                        except:
                            continue
            except:
                pass
        except ImportError:
            detected_network['metadata_check'] = 'requests not installed'
        
        is_colab = confidence > 0.3
        confidence = min(confidence, 1.0)
        
        return is_colab, confidence, detected_network
    
    def _check_runtime_info(self) -> Tuple[bool, float, Dict]:
        """Проверка информации о рантайме"""
        runtime_info = {}
        confidence = 0.0
        
        # Проверка версии Python
        python_version = platform.python_version()
        runtime_info['python_version'] = python_version
        
        # Colab обычно использует Python 3.x
        if python_version.startswith('3.'):
            confidence += 0.1
        
        # Проверка процесса
        try:
            with open('/proc/self/cgroup', 'r') as f:
                cgroup_content = f.read()
                runtime_info['cgroup'] = cgroup_content[:500]
                
                if 'docker' in cgroup_content or 'gcontainers' in cgroup_content:
                    confidence += 0.3
                    
                if 'colab' in cgroup_content.lower():
                    confidence += 0.4
        except:
            pass
        
        # Проверка используемой памяти
        try:
            import psutil
            mem = psutil.virtual_memory()
            runtime_info['total_memory_gb'] = mem.total / (1024**3)
            
            # Colab обычно имеет 12-25 GB RAM
            if 12 <= runtime_info['total_memory_gb'] <= 25:
                confidence += 0.2
        except ImportError:
            runtime_info['memory_check'] = 'psutil not installed'
        
        # Проверка CPU
        runtime_info['cpu_count'] = os.cpu_count()
        
        # Colab обычно имеет 2-4 CPU
        if runtime_info['cpu_count'] in [2, 4]:
            confidence += 0.1
        
        is_colab = confidence > 0.5
        confidence = min(confidence, 1.0)
        
        return is_colab, confidence, runtime_info
    
    def _check_gpu_tpu(self) -> Tuple[bool, float, Dict]:
        """Проверка наличия GPU/TPU"""
        gpu_tpu_info = {}
        confidence = 0.0
        
        # Проверка GPU
        if self.enable_gpu_check:
            gpu_detected = False
            
            # Способ 1: через nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    gpu_tpu_info['nvidia_smi'] = result.stdout[:500]
                    gpu_detected = True
                    confidence += 0.4
            except:
                pass
            
            # Способ 2: через torch
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_tpu_info['torch_cuda'] = {
                        'device_count': torch.cuda.device_count(),
                        'current_device': torch.cuda.current_device(),
                        'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                    }
                    gpu_detected = True
                    confidence += 0.3
            except ImportError:
                pass
            
            # Способ 3: через tensorflow
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    gpu_tpu_info['tensorflow_gpu'] = [str(gpu) for gpu in gpus]
                    gpu_detected = True
                    confidence += 0.3
            except ImportError:
                pass
            
            if gpu_detected:
                gpu_tpu_info['gpu_detected'] = True
        
        # Проверка TPU
        if self.enable_tpu_check:
            try:
                # Прямая проверка переменных TPU
                tpu_env_vars = ['COLAB_TPU_ADDR', 'TPU_NAME']
                for var in tpu_env_vars:
                    if var in os.environ:
                        gpu_tpu_info[f'tpu_env_{var}'] = os.environ[var]
                        confidence += 0.4
                
                # Проверка через tensorflow
                try:
                    import tensorflow as tf
                    tpus = tf.config.list_physical_devices('TPU')
                    if tpus:
                        gpu_tpu_info['tpu_devices'] = [str(tpu) for tpu in tpus]
                        confidence += 0.5
                except:
                    pass
            except:
                pass
        
        is_colab = confidence > 0.3
        confidence = min(confidence, 1.0)
        
        return is_colab, confidence, gpu_tpu_info
    
    def _check_custom_indicators(self) -> Tuple[bool, float, Dict]:
        """Кастомные проверки и эвристики"""
        custom_info = {}
        confidence = 0.0
        
        # Проверка времени жизни процесса (Colab сессии обычно недолгие)
        try:
            import time
            import psutil
            process = psutil.Process()
            create_time = process.create_time()
            uptime = time.time() - create_time
            
            custom_info['process_uptime_hours'] = uptime / 3600
            
            # Если процесс работает менее 24 часов - возможный признак Colab
            if uptime < 86400:  # 24 часа
                confidence += 0.1
        except:
            pass
        
        # Проверка размера /tmp (в Colab обычно много места)
        try:
            import shutil
            tmp_usage = shutil.disk_usage('/tmp')
            custom_info['tmp_free_gb'] = tmp_usage.free / (1024**3)
            
            if tmp_usage.free > 10 * (1024**3):  # Более 10 GB свободно
                confidence += 0.15
        except:
            pass
        
        # Проверка наличия специфичных процессов
        try:
            colab_processes = ['jupyter-notebook', 'python3', 'python']
            running_processes = []
            
            for proc in psutil.process_iter(['name']):
                try:
                    running_processes.append(proc.info['name'])
                except:
                    continue
            
            colab_running = any(p in running_processes for p in colab_processes)
            if colab_running:
                custom_info['colab_processes'] = True
                confidence += 0.2
        except:
            pass
        
        # Проверка через наличие Jupyter ноутбука
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython:
                custom_info['ipython_available'] = True
                
                # Получение информации о ядре
                if hasattr(ipython, 'kernel'):
                    custom_info['kernel_info'] = str(type(ipython.kernel))
                    confidence += 0.1
        except:
            pass
        
        is_colab = confidence > 0.2
        confidence = min(confidence, 1.0)
        
        return is_colab, confidence, custom_info
    
    def detect(self, verbose: bool = False) -> ColabDetectionResult:
        """
        Основной метод детектирования Colab
        
        Args:
            verbose: Выводить подробную информацию
            
        Returns:
            ColabDetectionResult: Результат детектирования
        """
        detection_methods = {}
        all_details = {}
        total_confidence = 0.0
        method_count = 0
        warnings_list = []
        
        # Выполнение всех проверок
        checks = [
            (ColabDetectionMethod.ENV_VARIABLES, self._check_env_variables),
            (ColabDetectionMethod.PYTHON_MODULES, self._check_python_modules),
            (ColabDetectionMethod.FILESYSTEM, self._check_filesystem),
            (ColabDetectionMethod.NETWORK, self._check_network),
            (ColabDetectionMethod.RUNTIME_INFO, self._check_runtime_info),
            (ColabDetectionMethod.GPU_INFO, self._check_gpu_tpu),
            (ColabDetectionMethod.CUSTOM_CHECKS, self._check_custom_indicators),
        ]
        
        for method_name, check_func in checks:
            try:
                is_colab, confidence, details = check_func()
                detection_methods[method_name] = is_colab
                all_details[method_name.value] = details
                
                if is_colab:
                    total_confidence += confidence
                    method_count += 1
                    
                    if verbose:
                        print(f"[✓] {method_name.value}: detected (confidence: {confidence:.2f})")
                else:
                    if verbose:
                        print(f"[ ] {method_name.value}: not detected")
            except Exception as e:
                warning_msg = f"Error in {method_name.value}: {str(e)}"
                warnings_list.append(warning_msg)
                if verbose:
                    print(f"[!] {method_name.value}: error - {str(e)}")
                detection_methods[method_name] = False
                all_details[method_name.value] = {'error': str(e)}
        
        # Расчет общей уверенности
        if method_count > 0:
            avg_confidence = total_confidence / method_count
        else:
            avg_confidence = 0.0
        
        # Определение типа рантайма
        runtime_type = None
        gpu_details = all_details.get('gpu_info', {})
        if gpu_details.get('tpu_devices') or any('tpu' in k for k in gpu_details):
            runtime_type = 'TPU'
        elif gpu_details.get('gpu_detected'):
            runtime_type = 'GPU'
        else:
            runtime_type = 'CPU'
        
        # Определение версии Colab
        colab_version = None
        env_details = all_details.get('env_variables', {})
        if 'COLAB_RELEASE_TAG' in env_details:
            colab_version = env_details['COLAB_RELEASE_TAG']
        
        # Финальное решение
        final_is_colab = avg_confidence > 0.5
        
        result = ColabDetectionResult(
            is_colab=final_is_colab,
            confidence=round(avg_confidence, 3),
            detection_methods=detection_methods,
            colab_version=colab_version,
            runtime_type=runtime_type,
            colab_details=all_details,
            warnings=warnings_list
        )
        
        self._cache['last_result'] = result
        
        return result
    
    def get_detailed_report(self) -> str:
        """Получить подробный отчет в виде строки"""
        if 'last_result' not in self._cache:
            self.detect()
        
        result = self._cache['last_result']
        
        report_lines = [
            "=" * 60,
            "GOOGLE COLAB DETECTION REPORT",
            "=" * 60,
            f"Colab Detected: {'YES' if result.is_colab else 'NO'}",
            f"Confidence: {result.confidence:.1%}",
            f"Colab Version: {result.colab_version or 'Unknown'}",
            f"Runtime Type: {result.runtime_type}",
            "",
            "Detection Methods:",
        ]
        
        for method, detected in result.detection_methods.items():
            status = "✓" if detected else "✗"
            report_lines.append(f"  {status} {method.value}")
        
        report_lines.append("")
        report_lines.append("Detailed Information:")
        
        for category, details in result.colab_details.items():
            report_lines.append(f"\n{category.upper()}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    report_lines.append(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            else:
                report_lines.append(f"  {str(details)[:200]}")
        
        if result.warnings:
            report_lines.append("\nWarnings:")
            for warning in result.warnings:
                report_lines.append(f"  ⚠ {warning}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_report(self, filepath: str = "colab_detection_report.json"):
        """Сохранить отчет в JSON файл"""
        if 'last_result' not in self._cache:
            self.detect()
        
        result = self._cache['last_result']
        
        # Конвертация enum в строки для JSON
        report_data = {
            'is_colab': result.is_colab,
            'confidence': result.confidence,
            'colab_version': result.colab_version,
            'runtime_type': result.runtime_type,
            'detection_methods': {k.value: v for k, v in result.detection_methods.items()},
            'colab_details': result.colab_details,
            'warnings': result.warnings,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'hostname': socket.gethostname() if hasattr(socket, 'gethostname') else 'unknown'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return filepath


# Декоратор для автоматической проверки Colab
def colab_only(func):
    """Декоратор для выполнения функции только в Colab"""
    def wrapper(*args, **kwargs):
        detector = ColabDetector()
        result = detector.detect()
        
        if not result.is_colab:
            raise RuntimeError(
                f"Функция {func.__name__} может быть выполнена только в Google Colab.\n"
                f"Детектирование показало уверенность {result.confidence:.1%}."
            )
        
        print(f"✓ Работаем в Google Colab (уверенность: {result.confidence:.1%})")
        return func(*args, **kwargs)
    
    return wrapper


# Простая функция для быстрой проверки
def is_colab(threshold: float = 0.5) -> bool:
    """
    Быстрая проверка на Colab
    
    Args:
        threshold: Порог уверенности (0.0-1.0)
    
    Returns:
        bool: True если похоже на Colab
    """
    detector = ColabDetector(enable_network_checks=False)
    result = detector.detect(verbose=False)
    return result.is_colab and result.confidence >= threshold


# Утилитарные функции
def get_colab_info() -> Optional[Dict]:
    """Получить информацию о Colab если мы в нем"""
    if not is_colab():
        return None
    
    detector = ColabDetector()
    result = detector.detect()
    
    info = {
        'is_colab': result.is_colab,
        'confidence': result.confidence,
        'version': result.colab_version,
        'runtime': result.runtime_type,
        'gpu_available': result.runtime_type == 'GPU',
        'tpu_available': result.runtime_type == 'TPU',
    }
    
    return info


def colab_setup_hook():
    """Хук для автоматической настройки в Colab"""
    if not is_colab():
        return False
    
    print("🔧 Настройка среды Colab...")
    
    # Установка полезных пакетов если нужно
    try:
        import IPython
        from IPython.display import display, HTML
        
        # Добавление CSS для красивого отображения
        display(HTML("""
        <style>
        .colab-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """))
        
        detector = ColabDetector()
        result = detector.detect()
        
        info_html = f"""
        <div class="colab-info">
            <h3>🎉 Google Colab Detected</h3>
            <p><strong>Version:</strong> {result.colab_version or 'Unknown'}</p>
            <p><strong>Runtime:</strong> {result.runtime_type}</p>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p><strong>Python:</strong> {platform.python_version()}</p>
        </div>
        """
        display(HTML(info_html))
        
    except:
        pass
    
    return True


def full_check_is_colab():
    
    detector = ColabDetector(
        enable_network_checks=True,
        enable_gpu_check=True,
        enable_tpu_check=True
    )
    
    result = detector.detect(verbose=False)
    
    return result.is_colab
