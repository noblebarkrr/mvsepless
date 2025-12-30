import platform
import sys

def easy_check_is_colab() -> bool:
    if platform.machine() == "x86_64" and "Linux" in platform.platform():
        try:
            import google.colab
            module_path: str = google.colab.__file__
            if module_path.startswith("/usr/local/lib/python") and module_path.endswith("/dist-packages/google/colab/__init__.py"):
                return True
            else:
                return False
        except ImportError:
            return False
    else:
        return False
