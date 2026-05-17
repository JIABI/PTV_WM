import json
from pathlib import Path

def write_manifest(outroot):
    p=Path(outroot)/'manifest.json'
    files=[str(x.relative_to(Path(outroot))) for x in Path(outroot).rglob('*') if x.is_file()]
    p.write_text(json.dumps({'files':sorted(files)},indent=2),encoding='utf-8')
