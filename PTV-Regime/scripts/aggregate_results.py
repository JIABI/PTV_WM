import argparse
import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from licwm.reporting.aggregators import aggregate_runs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', action='append', default=None, help='Section csv stem or filename to aggregate, repeatable.')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    sections = None
    if args.section:
        sections = [s if s.endswith('.csv') else f'{s}.csv' for s in args.section]
    aggregate_runs(sections=sections, verbose=args.verbose)
