# Final display contract for paper artifacts

This document fixes the presentation-layer contract used when exporting Main Figure/Table artifacts and their SI counterparts.

## Method ordering

All tables/figures that include model or ablation rows should use this order when the field is available:

1. LIC-WM
2. AutoPhysiCK-WM
3. Context-WM
4. GRU-WM
5. Transformer-WM
6. CfC-WM
7. MoE-WM
8. No Climate
9. No Jump
10. No Slow Drift
11. Leakage
12. Bounded Residual
13. No Event Token
14. Small Bank
15. Medium Bank
16. Large Bank
17. GT Climate Oracle
18. Law-Parameter Oracle

## Domain ordering

1. LIC-Boids
2. ETH/UCY
3. SDD
4. INTERACTION
5. LIC-UAV

## Main artifact display intent

- **Table 1** presents benchmark/protocol setup, sorted by domain then task.
- **Table 2** presents LIC-Boids mechanism evidence, sorted by task then method.
- **Table 3** presents cross-domain crowd/traffic predictive stability, sorted by domain, scene, then method.
- **Table 4** presents LIC-UAV control value, sorted by task then method.
- **Table 5** presents robustness audits and stress tests, sorted by method.

- **Figure 2** should foreground mechanism fields: geometry match quality, law-state separation, response separation, monotonicity.
- **Figure 3** should foreground predictive stability fields: step RMSE, rollout ADE/FDE, rollout horizon.
- **Figure 4** should foreground control fields: success rate, safety violation, recovery latency, CVaR.
- **Figure 5** should foreground defense fields: law-state TV, HF-ratio, scale-transfer gap, rollout horizon.

## Caption discipline

Exported caption stubs are intentionally short and should be refined in the paper source, but they are already aligned to the intended role of each artifact.
