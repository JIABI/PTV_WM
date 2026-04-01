# Method mapping
- `q_phi(c_t|H_t)`: `models/common/sequence_encoders.py::HistoryEncoder`, `models/licwm/climate_encoder.py`
- `ell_t=T_psi(c_t)=(rho,beta,tau)`: `models/licwm/law_state_head.py`
- local prototype response: `models/licwm/channel_bank.py`, `models/licwm/basis_families.py`
- message equation with `chi, Gamma, beta`: `models/licwm/message_passing.py`
- fast dynamics `U_theta`: `models/licwm/fast_state_update.py`
- slow/jump climate dynamics: `models/licwm/climate_transition.py`
- anti-steganography TV/HF: `metrics/antisteg.py`, `evaluation/antisteg.py`
