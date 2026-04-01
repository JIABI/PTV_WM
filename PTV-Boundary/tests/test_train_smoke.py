import subprocess

def test_train_smoke():
    subprocess.run(["python", "training/train_audit.py", "env=dummy", "runtime.max_steps=1"], check=True)
    subprocess.run(["python", "training/train_ralag_wm.py", "env=dummy", "runtime.max_steps=1"], check=True)
    subprocess.run(["python", "training/train_baseline.py", "env=dummy", "runtime.max_steps=1"], check=True)
