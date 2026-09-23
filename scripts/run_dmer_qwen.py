"""Readiness by default; --execute runs authorized local-snapshot Qwen 5+5 training."""
import argparse
import json
from pathlib import Path

from sc_mv_dmer.training.dmer_qwen import QwenInputs, dry_run, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in QwenInputs.__dataclass_fields__:
        parser.add_argument('--' + name.replace('_','-'), type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicitly run after every readiness check passes')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=52826381)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=16)
    parser.add_argument('--skip-validation-generation', action='store_true')
    args = parser.parse_args()
    inputs = QwenInputs(**{name:getattr(args,name) for name in QwenInputs.__dataclass_fields__})
    if args.execute:
        if args.output is None:
            parser.error('--execute requires --output')
        result = run_training(inputs,args.output,seed=args.seed,device=args.device,
            micro_batch=args.micro_batch,accumulation=args.accumulation,
            generate_explanations=not args.skip_validation_generation)
    else:
        result = dry_run(inputs)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
