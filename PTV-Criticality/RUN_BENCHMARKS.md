# Benchmark commands

## DMC

```bash
pip install -e .[dmc]
bash scripts/run_dmc_3tasks.sh
```

Single-task example:

```bash
python -m fatewm.experiments.dmc.train \
  env=dmc_walker_walk \
  method=scheme1_dmc \
  algo=minimal \
  device=cuda
```

## Procgen

```bash
pip install -e .[procgen]
bash scripts/run_procgen_3envs.sh
```

Single-environment example:

```bash
python -m fatewm.experiments.procgen.train \
  env=procgen_coinrun \
  method=scheme1_procgen \
  algo=minimal \
  device=cuda
```

## Atari-100k style discrete benchmark

```bash
pip install -e .[atari]
bash scripts/run_atari100k_3games.sh
```

Single-game example:

```bash
python -m fatewm.experiments.atari100k.train \
  env=atari100k_breakout \
  method=rrrm \
  algo=minimal \
  device=cuda
```
