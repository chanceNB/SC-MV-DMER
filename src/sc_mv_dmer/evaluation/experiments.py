"""Frozen-checkpoint held-out evaluation and fixed-split five-seed reporting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sc_mv_dmer.data.deam_dmer_full import file_sha256
from sc_mv_dmer.evaluation.dmer_metrics import evaluate_songs
from sc_mv_dmer.training.dmer_acoustic import SEEDS, TrainNormalizer
from sc_mv_dmer.training.dmer_runner import WindowDataset, load_rows, predict, save_predictions, stitch_song_predictions


def check_frozen_run(spec: dict, final: dict) -> None:
    if spec.get("mode") != "formal":
        raise ValueError("held-out evaluation requires a formal frozen source run")
    if final.get("status") != "SUCCEEDED":
        raise ValueError("source training run must be SUCCEEDED")


def aggregate_five_seeds(rows: list[dict]) -> dict:
    if len(rows) != 5 or {r["spec"]["seed"] for r in rows} != set(SEEDS):
        raise ValueError("aggregation requires exactly the registered five unique seeds")
    if len({r["spec"]["split_sha256"] for r in rows}) != 1:
        raise ValueError("cannot aggregate different split memberships")
    if len({r["spec"]["variant"] for r in rows}) != 1:
        raise ValueError("cannot aggregate different model variants")
    if any(r["spec"].get("mode") != "formal" for r in rows):
        raise ValueError("formal five-seed report cannot include development runs")
    # Exclude only recorded runtime choices and seed. Research settings must match.
    runtime = {"seed", "batch_size", "device", "git_status"}
    signatures = {json.dumps({k:v for k,v in r["spec"].items() if k not in runtime}, sort_keys=True) for r in rows}
    if len(signatures) != 1:
        raise ValueError("five seeds must share the same research specification")
    result = {"seed_count":5,"seeds":list(SEEDS),"std_ddof":1,"split_sha256":rows[0]["spec"]["split_sha256"],
              "variant":rows[0]["spec"]["variant"],"macro":{}}
    for dimension in ("valence","arousal"):
        result["macro"][dimension]={}
        for metric in ("ccc","pcc","rmse"):
            values=[r["metrics"]["macro"][dimension][metric] for r in rows]
            if any(v is None or not np.isfinite(v) for v in values):
                raise ValueError("invalid metric cannot be omitted from seed aggregation")
            result["macro"][dimension][metric]={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"values":values}
    return result


def evaluate_checkpoint(*, source:Path, manifest_path:Path, split_path:Path,
                        targets_path:Path, cache_path:Path|None, role:str, output:Path,
                        device:str="cuda", batch_size:int=16)->dict:
    if role not in ("test","long_test"):
        raise ValueError("held-out role must be test or long_test")
    if output.exists(): raise FileExistsError("evaluation output is immutable")
    spec=json.loads((source/"run_spec.json").read_text(encoding="utf-8"))
    final=json.loads((source/"final.json").read_text(encoding="utf-8"))
    check_frozen_run(spec,final)
    manifest,split,rows=load_rows(manifest_path,split_path,targets_path,cache_path)
    if spec["split_sha256"]!=split["split_sha256"] or spec["dataset_manifest_sha256"]!=manifest["manifest_sha256"]:
        raise ValueError("evaluation dataset differs from training registration")
    if spec["feature_manifest_file_sha256"]!=(file_sha256(cache_path) if cache_path else None):
        raise ValueError("evaluation features differ from training registration")
    rows=[r for r in rows if r["split"]==role]
    if spec["variant"]=="constant":
        constant=json.loads((source/"constant.json").read_text())["value"]
        songs=stitch_song_predictions([dict(r,prediction=np.broadcast_to(constant,(60,2))) for r in rows])
        metrics=evaluate_songs(songs)
        checkpoint_hash=file_sha256(source/"constant.json")
    else:
        from sc_mv_dmer.models import AcousticModel
        data=json.loads((source/"normalization.json").read_text())
        if data.get("fit_role")!="train" or data.get("split_sha256")!=split["split_sha256"]:
            raise ValueError("normalization provenance is not train-only")
        normalizer=TrainNormalizer({k:np.asarray(v) for k,v in data["mean"].items()},
                                   {k:np.asarray(v) for k,v in data["std"].items()})
        loader=DataLoader(WindowDataset(rows,normalizer,spec["raw_mert"]),batch_size=batch_size,shuffle=False)
        model=AcousticModel(variant=spec["variant"],hidden_dim=spec["hidden_dim"],use_timesnet=spec["use_timesnet"]).to(device)
        checkpoint=torch.load(source/"best.pt",map_location=device,weights_only=False)
        if checkpoint["spec"]!=spec or checkpoint["epoch"]!=final["best_epoch"]:
            raise ValueError("checkpoint does not match frozen source manifest")
        model.load_state_dict(checkpoint["state_dict"])
        metrics,songs=predict(model,loader,torch.device(device))
        checkpoint_hash=file_sha256(source/"best.pt")
    output.mkdir(parents=True)
    report={"status":"SUCCEEDED","role":role,"source_run_spec_sha256":file_sha256(source/"run_spec.json"),
            "source_checkpoint_sha256":checkpoint_hash,"metrics":metrics,"fit_performed":False}
    (output/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save_predictions(output/"predictions.jsonl",songs)
    return report
