import argparse
from pathlib import Path

from sc_mv_dmer.evaluation.experiments import evaluate_checkpoint


if __name__=="__main__":
    p=argparse.ArgumentParser(description="Evaluate a frozen successful formal acoustic run; never fits on test data.")
    for name in ("source","manifest","split","targets","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cache",type=Path)
    p.add_argument("--role",choices=["test","long_test"],required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=16)
    a=p.parse_args()
    r=evaluate_checkpoint(source=a.source,manifest_path=a.manifest,split_path=a.split,targets_path=a.targets,
                          cache_path=a.cache,role=a.role,output=a.output,device=a.device,batch_size=a.batch_size)
    print({"status":r["status"],"role":r["role"],"macro":r["metrics"]["macro"]})
