"""Rebuild ordinary DMER labels without excluding any DEAM song."""
import argparse
import json
from pathlib import Path

from sc_mv_dmer.data.deam_dmer_full import VERSION, materialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    output = args.data_root / 'processed' / VERSION
    manifest_path = args.repo_root / 'manifests/datasets/deam-dmer-full-v1.json'
    audit_path = args.repo_root / 'reports/data/deam-dmer-full-v1-audit.json'
    for path in (output, manifest_path, audit_path):
        if path.exists():
            raise FileExistsError(f'refusing to overwrite: {path}')
    result = materialize(args.data_root, args.repo_root / 'manifests/datasets/deam-pdmer-clean-v3.json',
                         output, progress=lambda message: print(message, flush=True))
    for source, destination in ((output / 'manifest.json', manifest_path), (output / 'audit.json', audit_path)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as f:
            f.write(source.read_bytes())
    print(json.dumps({'manifest': result['manifest_id'], 'population': result['population'],
                      'output': str(output), 'formal_training_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
