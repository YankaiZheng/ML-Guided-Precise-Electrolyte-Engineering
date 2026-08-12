#!/usr/bin/env python3
"""Predict the frozen xTB-vector D model on unlabeled Domain65k structures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import run_domain65k_d_vector_equivariant as vector


class UnlabeledVectorDataset(Dataset):
    def __init__(self, meta: pd.DataFrame):
        self.meta = meta.reset_index(drop=True)
        self.store = vector.RawStore()

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, item: int) -> dict:
        row = self.meta.iloc[item]
        raw = self.store.get(int(row.domain_index))
        conformers = raw.get("conformers") or []
        if raw.get("status") != "ok" or len(conformers) != 3:
            raise RuntimeError(f"Incomplete xTB graph at domain_index={row.domain_index}")
        energies = np.asarray([float(conf["energy_eh"]) for conf in conformers], dtype=float)
        if not np.isfinite(energies).all():
            raise RuntimeError(f"Non-finite xTB conformer energy at domain_index={row.domain_index}")
        # No DFT geometry/label is needed for inference; the lowest xTB-energy
        # conformer is the same deployable coordinate anchor used by training.
        return {
            "domain_index": int(row.domain_index),
            "D": 0.0,
            "target_mu": np.zeros(3, dtype=np.float32),
            "rank_target": 0.0,
            "anchor": int(np.argmin(energies)),
            "raw": raw,
        }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv)
    required = {"domain_index", "canonical_smiles"}
    if not required.issubset(source.columns):
        raise RuntimeError(f"Inference input is missing {sorted(required - set(source.columns))}")
    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    model = vector.XTBVectorDipole(
        hidden=int(model_args.get("hidden", 128)),
        layers=int(model_args.get("layers", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    # QMe14S contains two duplicated source rows that map to the same feature
    # identity. Predict that identity once and retain both locked records when
    # the prediction is joined back below.
    unique_source = source.drop_duplicates("domain_index", keep="first").reset_index(drop=True)
    data = UnlabeledVectorDataset(unique_source)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=vector.collate,
    )
    prediction = vector.predict(model, loader, device).drop(columns=["D"])
    output = source.merge(prediction, on="domain_index", how="left", validate="many_to_one")
    if len(output) != len(source) or output.filter(like="pred__").isna().any().any():
        raise RuntimeError("Vector inference produced incomplete predictions")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print({"n_predicted": len(output), "device": str(device), "output": args.output_csv}, flush=True)


if __name__ == "__main__":
    main()
