"""Allowlisted source bundle; never include credentials, unrelated repos or raw data."""
import argparse
import hashlib
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(output, 'w:gz') as archive:
    for name in ('rmfsr', 'configs', 'tests', 'scripts', 'README.md', 'TRAINING.md', 'ARCHITECTURE.md', 'architecture-audit.json', 'pyproject.toml'):
        path = root / name
        for entry in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if entry.is_file() and not entry.is_symlink() and '__pycache__' not in entry.parts and entry.suffix != '.pyc':
                archive.add(entry, arcname=str(Path('rmfsr') / entry.relative_to(root)))
sha = hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix(output.suffix + '.sha256').write_text(f'{sha}  {output.name}\n')
print(output, output.stat().st_size, 'bytes', sha)
