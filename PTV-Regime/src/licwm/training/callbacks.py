def on_epoch_end(epoch: int, metrics: dict):
    print(f"[epoch {epoch}] total={metrics['total']:.4f} step={metrics['step']:.4f} roll={metrics['roll']:.4f} slow={metrics['slow']:.4f} jump={metrics['jump']:.4f}")
