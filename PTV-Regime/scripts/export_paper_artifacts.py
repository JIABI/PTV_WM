from __future__ import annotations

import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from licwm.reporting.figures import make_main_figures, make_si_figures
from licwm.reporting.tables import make_main_tables, make_si_tables
from licwm.reporting.validate import validate_artifacts
from licwm.reporting.specs import ALL_FIGURE_SPECS, ALL_TABLE_SPECS


def write_manifest_summary(out_path: str = 'outputs/tables/paper_artifact_index.json') -> None:
    payload = {
        'tables': [
            {
                'artifact_id': s.artifact_id,
                'title': s.title,
                'source_csv': s.source_csv,
                'output_stem': s.output_stem,
                'preferred_columns': list(s.preferred_columns),
                'required_columns': list(s.required_columns),
                'caption_stub': s.caption_stub,
            }
            for s in ALL_TABLE_SPECS
        ],
        'figures': [
            {
                'artifact_id': s.artifact_id,
                'title': s.title,
                'source_csv': s.source_csv,
                'output_stem': s.output_stem,
                'preferred_columns': list(s.preferred_columns),
                'required_columns': list(s.required_columns),
                'generated': s.generated,
                'caption_stub': s.caption_stub,
            }
            for s in ALL_FIGURE_SPECS
        ],
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    make_main_tables()
    make_si_tables()
    make_main_figures()
    make_si_figures()
    validate_artifacts()
    write_manifest_summary()
