"""Run corrected full-population sensor development; no formal gate auto-pass."""
import argparse
import json
from pathlib import Path

from sc_mv_dmer.training.dmer_sensor import run_sensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'split', 'targets', 'cache', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed', type=int, default=52826381)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--write-cot-candidates', action='store_true', help='Write unreviewed train-only templates and blank 100-item review sheet')
    args = parser.parse_args()
    result = run_sensor(manifest_path=args.manifest, split_path=args.split, targets_path=args.targets,
                        cache_path=args.cache, output=args.output, seed=args.seed, epochs=args.epochs,
                        patience=args.patience, batch_size=args.batch_size, lr=args.lr, device=args.device,
                        write_cot=args.write_cot_candidates)
    print(json.dumps({'status': result['status'], 'formal_pass': result['formal_pass'], 'gate': result['gate']}, indent=2))


if __name__ == '__main__': main()
