import json
import os
import platform

def log_environment(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/env.json", "w", encoding="utf-8") as f:
        json.dump({"platform": platform.platform()}, f, indent=2)
