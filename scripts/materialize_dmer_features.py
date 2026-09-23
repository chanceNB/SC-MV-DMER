"""Build resumable full-population DMER features without modifying source audio."""
import argparse
import json
from pathlib import Path

from sc_mv_dmer.features.dmer_full import DEFAULT_MODEL_ROOT, materialize_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--decoded-audit', type=Path, required=True,
                        help='immutable decoded-PCM audit; source of actual input support masks')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='Process first N songs; manifest remains incomplete until all 1802')
    parser.add_argument('--handcrafted-only', action='store_true', help='Diagnostic cache; never qualifies as four-view cache')
    parser.add_argument('--mert-model-root', type=Path, default=DEFAULT_MODEL_ROOT)
    args = parser.parse_args()
    result = materialize_features(args.manifest, args.data_root, args.output_root, device=args.device,
                                  limit=args.limit, handcrafted_only=args.handcrafted_only,
                                  mert_model_root=args.mert_model_root,
                                  decoded_audit_path=args.decoded_audit,
                                  progress=lambda line: print(line, flush=True))
    print(json.dumps({k: result[k] for k in ('feature_version', 'completed_song_count', 'completed_window_count',
                                             'full_population_ready', 'four_view_ready')}, indent=2))


if __name__ == '__main__':
    main()
