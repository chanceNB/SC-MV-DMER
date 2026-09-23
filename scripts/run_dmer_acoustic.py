"""Explicit revised-DMER train/validation runner; test sets are never evaluated here."""
import argparse
import os
from pathlib import Path

from sc_mv_dmer.training.dmer_runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=["constant", "crnn", "mert_mlp", "concat", "bilstm_attention", "shared_state", "independent_state", "no_state_bias", "deep_only"])
    parser.add_argument("--seed", type=int, default=52826381)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="development", choices=["development", "formal"])
    parser.add_argument("--timesnet", action="store_true")
    parser.add_argument("--raw-mert", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    result = run_experiment(manifest_path=args.manifest, split_path=args.split, targets_path=args.targets,
                            cache_path=args.cache, output=args.output, variant=args.variant, seed=args.seed,
                            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                            device=args.device, lr=args.lr, hidden_dim=args.hidden_dim,
                            use_timesnet=args.timesnet, raw_mert=args.raw_mert, mode=args.mode)
    print(result)


if __name__ == "__main__":
    main()
