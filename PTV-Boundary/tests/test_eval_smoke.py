import subprocess


def test_eval_smoke():
    subprocess.run(["python", "testing/eval_main_benchmark.py", "env=dummy"], check=True)
    subprocess.run(["python", "testing/eval_matched_fidelity.py", "env=dummy"], check=True)
    subprocess.run(["python", "testing/eval_oracle_substitution.py", "env=dummy"], check=True)
    subprocess.run(["python", "testing/eval_robustness.py", "env=dummy"], check=True)
    subprocess.run(["python", "testing/eval_frontier.py", "env=dummy"], check=True)
