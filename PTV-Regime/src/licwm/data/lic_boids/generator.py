from .simulator import LICBoidsSimulator

def generate_sequences(n_samples: int, n_agents: int, mode: str, horizon: int, seed: int = 0):
    sim = LICBoidsSimulator(n_agents=n_agents, mode=mode, seed=seed)
    for i in range(n_samples):
        yield sim.rollout(horizon)
