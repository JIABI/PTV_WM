import json
from pathlib import Path


def write_manifest(outroot, data_status):
    p = Path(outroot) / 'manifest.json'
    files = [str(x.relative_to(Path(outroot))) for x in Path(outroot).rglob('*') if x.is_file() and x.name != 'manifest.json']
    payload = {
        'files': sorted(files),
        'data_status': data_status,
        'manuscript_reproducible': data_status == 'manuscript',
    }
    p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
