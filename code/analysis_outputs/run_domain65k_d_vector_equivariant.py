#!/usr/bin/env python3
"""Vector-supervised, xTB-deployable equivariant dipole model.

The reference OPT geometry is consumed only by the prior alignment-label build.
This trainer reads xTB graphs/conformers plus targets already expressed in the
xTB anchor frame.  It therefore has exactly the same deployable inputs at
inference as the existing atom3d branch.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


REPOSITORY = Path(__file__).resolve().parents[2]
STUDY_ARTIFACT_ROOT = Path(__import__("os").environ.get("DP_COMPASS_STUDY_ARTIFACT_ROOT", REPOSITORY / "study_artifacts"))
DOMAIN = STUDY_ARTIFACT_ROOT / "domain65k"
RAW = DOMAIN / "features_v2_atom3d/chunks"
CV = DOMAIN / "domain65k_cv_pool_complete_features.csv"
ALIGN = DOMAIN / "features_v2_vector_alignment/vector_alignment_dev.pkl"
ALIGN_MANIFEST = DOMAIN / "features_v2_vector_alignment/vector_alignment_manifest.json"
EXCLUSION = STUDY_ARTIFACT_ROOT / "candidate_curation_v3/domain65k_candidate100_exact_exclusion.csv"
RUN = DOMAIN / "model_runs"
SEED = 20260714
KBT_EH = 3.166811563e-6 * 298.15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kabsch_rotation(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 3:
        mask = np.ones(len(source), dtype=bool)
    src = source[mask] - source[mask].mean(axis=0)
    dst = target[mask] - target[mask].mean(axis=0)
    u, _, vt = np.linalg.svd(src.T @ dst)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation.astype(np.float32)


def radius_edges(pos: np.ndarray, cutoff: float = 6.0, max_neighbors: int = 32) -> np.ndarray:
    distance = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    edges: list[tuple[int, int]] = []
    for target in range(len(pos)):
        source = np.flatnonzero((distance[:, target] <= cutoff) & (distance[:, target] > 1e-7))
        if len(source) > max_neighbors:
            source = source[np.argsort(distance[source, target])[:max_neighbors]]
        edges.extend((int(item), target) for item in source)
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


class RawStore:
    @lru_cache(maxsize=16)
    def chunk(self, start: int) -> dict[int, dict]:
        path = RAW / f"chunk_{start:06d}_{min(start + 100, 65126):06d}.pkl"
        frame = pd.read_pickle(path)
        return {int(row["domain_index"]): row for row in frame.to_dict("records")}

    def get(self, domain_index: int) -> dict:
        return self.chunk((int(domain_index) // 100) * 100)[int(domain_index)]


class VectorDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, train_rank_reference: np.ndarray | None = None):
        self.meta = meta.reset_index(drop=True)
        self.store = RawStore()
        if train_rank_reference is None:
            self.rank = np.zeros(len(self.meta), dtype=np.float32)
        else:
            rank_map = pd.Series(train_rank_reference).rank(method="average", pct=True).to_numpy(np.float32)
            self.rank = rank_map

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, item: int) -> dict:
        row = self.meta.iloc[item]
        raw = self.store.get(int(row.domain_index))
        if raw["status"] != "ok" or len(raw["conformers"]) != 3:
            raise RuntimeError(f"Incomplete xTB graph at {row.domain_index}")
        return {
            "domain_index": int(row.domain_index),
            "D": float(row.D),
            "target_mu": np.asarray([row.target_mu_x, row.target_mu_y, row.target_mu_z], dtype=np.float32),
            "rank_target": float(self.rank[item]),
            "anchor": int(row.anchor_conformer),
            "raw": raw,
        }


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    z_all: list[np.ndarray] = []
    pos_all: list[np.ndarray] = []
    charge_all: list[np.ndarray] = []
    formal_all: list[np.ndarray] = []
    aromatic_all: list[np.ndarray] = []
    edge_all: list[np.ndarray] = []
    atom_conf: list[np.ndarray] = []
    conf_mol: list[int] = []
    baseline: list[np.ndarray] = []
    delta_energy: list[float] = []
    target_mu, target_d, rank_target, domain_index = [], [], [], []
    offset = 0
    conf_id = 0
    for mol_id, item in enumerate(items):
        graph = item["raw"]["graph"]
        z = np.asarray(graph["z"], dtype=np.int64)
        formal = np.asarray(graph["formal_charge"], dtype=np.float32)
        aromatic = np.asarray(graph["aromatic"], dtype=np.float32)
        conformers = item["raw"]["conformers"]
        anchor = int(item["anchor"])
        anchor_pos = np.asarray(conformers[anchor]["pos"], dtype=np.float32)
        energies = np.asarray([float(conf["energy_eh"]) for conf in conformers], dtype=np.float64)
        energy0 = float(np.nanmin(energies))
        for conformer in conformers:
            pos = np.asarray(conformer["pos"], dtype=np.float32)
            rotation = kabsch_rotation(pos, anchor_pos, z != 1)
            pos = (pos - pos.mean(axis=0, keepdims=True)) @ rotation
            q = np.asarray(conformer["charge"], dtype=np.float32)
            vec = np.asarray(conformer["full_dipole_vector_debye"], dtype=np.float32) @ rotation
            edges = radius_edges(pos) + offset
            z_all.append(z)
            pos_all.append(pos)
            charge_all.append(q)
            formal_all.append(formal)
            aromatic_all.append(aromatic)
            edge_all.append(edges)
            atom_conf.append(np.full(len(z), conf_id, dtype=np.int64))
            conf_mol.append(mol_id)
            baseline.append(vec)
            delta_energy.append(float(np.clip((float(conformer["energy_eh"]) - energy0) / KBT_EH, 0.0, 50.0)))
            offset += len(z)
            conf_id += 1
        target_mu.append(item["target_mu"])
        target_d.append(item["D"])
        rank_target.append(item["rank_target"])
        domain_index.append(item["domain_index"])
    return {
        "z": torch.from_numpy(np.concatenate(z_all)),
        "pos": torch.from_numpy(np.concatenate(pos_all)),
        "charge": torch.from_numpy(np.concatenate(charge_all)),
        "formal": torch.from_numpy(np.concatenate(formal_all)),
        "aromatic": torch.from_numpy(np.concatenate(aromatic_all)),
        "edge_index": torch.from_numpy(np.concatenate(edge_all, axis=1)),
        "atom_conf": torch.from_numpy(np.concatenate(atom_conf)),
        "conf_mol": torch.tensor(conf_mol, dtype=torch.long),
        "baseline_mu": torch.tensor(np.asarray(baseline), dtype=torch.float32),
        "delta_energy": torch.tensor(delta_energy, dtype=torch.float32),
        "target_mu": torch.tensor(np.asarray(target_mu), dtype=torch.float32),
        "target_d": torch.tensor(target_d, dtype=torch.float32),
        "rank_target": torch.tensor(rank_target, dtype=torch.float32),
        "domain_index": torch.tensor(domain_index, dtype=torch.long),
    }


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    result = values.new_zeros((size,) + values.shape[1:])
    result.index_add_(0, index, values)
    return result


def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    total = scatter_sum(values, index, size)
    count = values.new_zeros(size)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return total / count.clamp_min(1.0).reshape((size,) + (1,) * (values.ndim - 1))


def segment_softmax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    result = torch.empty_like(values)
    for group in range(size):
        mask = index == group
        result[mask] = torch.softmax(values[mask], dim=0)
    return result


def random_rotation(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.randn(3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    sign = torch.where(torch.diag(r) >= 0, torch.ones(3, device=device, dtype=dtype), -torch.ones(3, device=device, dtype=dtype))
    q = q @ torch.diag(sign)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


class EquivariantMessage(nn.Module):
    def __init__(self, hidden: int, radial: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(hidden * 2 + radial, hidden * 3), nn.SiLU(), nn.Linear(hidden * 3, hidden * 3))
        self.scalar_update = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))
        self.vector_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor, edge: torch.Tensor, unit: torch.Tensor, rbf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge[0], edge[1]
        message = self.message(torch.cat([scalar[source], scalar[target], rbf], dim=1))
        msg_scalar, msg_vector, msg_direction = message.chunk(3, dim=1)
        scalar_agg = scatter_sum(msg_scalar * scalar[source], target, scalar.size(0))
        vector_agg = scatter_sum(
            msg_vector.unsqueeze(-1) * vector[source] + msg_direction.unsqueeze(-1) * unit.unsqueeze(1),
            target,
            scalar.size(0),
        )
        vector_norm = torch.sqrt(torch.sum(vector_agg * vector_agg, dim=-1) + 1e-8)
        update = self.scalar_update(torch.cat([scalar, scalar_agg, vector_norm], dim=1))
        gate = self.vector_gate(torch.cat([scalar, update], dim=1)).unsqueeze(-1)
        return scalar + update, vector + gate * vector_agg


class XTBVectorDipole(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 4, radial: int = 24):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0.0, 6.0, radial), requires_grad=False)
        self.width = 1.8
        self.embedding = nn.Embedding(64, hidden)
        self.atom_input = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([EquivariantMessage(hidden, radial) for _ in range(layers)])
        self.delta_charge = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.local_coeff = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.attention = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.rank_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pos, edge = batch["pos"], batch["edge_index"]
        source, target = edge[0], edge[1]
        displacement = pos[source] - pos[target]
        distance = torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-6)
        unit = displacement / distance.unsqueeze(1)
        rbf = torch.exp(-self.width * (distance.unsqueeze(1) - self.centers.to(pos)) ** 2)
        scalar = self.embedding(batch["z"].clamp_max(63)) + self.atom_input(torch.stack([batch["charge"], batch["formal"], batch["aromatic"]], dim=1))
        vector = scalar.new_zeros((scalar.size(0), scalar.size(1), 3))
        for layer in self.layers:
            scalar, vector = layer(scalar, vector, edge, unit, rbf)
        n_conf = int(batch["conf_mol"].numel())
        atom_conf = batch["atom_conf"]
        q_delta = self.delta_charge(scalar).squeeze(1)
        q_delta = q_delta - scatter_mean(q_delta, atom_conf, n_conf)[atom_conf]
        local = torch.sum(self.local_coeff(scalar).unsqueeze(-1) * vector, dim=1)
        delta_mu = scatter_sum(q_delta.unsqueeze(1) * pos + local, atom_conf, n_conf)
        conf_mu = batch["baseline_mu"] + delta_mu
        conf_scalar = scatter_mean(scalar, atom_conf, n_conf)
        logits = self.attention(conf_scalar).squeeze(1) - batch["delta_energy"]
        n_mol = int(batch["target_d"].numel())
        weights = segment_softmax(logits, batch["conf_mol"], n_mol)
        molecular_mu = scatter_sum(conf_mu * weights.unsqueeze(1), batch["conf_mol"], n_mol)
        pooled_scalar = scatter_sum(conf_scalar * weights.unsqueeze(1), batch["conf_mol"], n_mol)
        return molecular_mu, self.rank_head(pooled_scalar).squeeze(1)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def pairwise_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = target[:, None] - target[None, :]
    mask = torch.triu(delta != 0, diagonal=1)
    if not mask.any():
        return score.new_tensor(0.0)
    return F.softplus(-torch.sign(delta[mask]) * (score[:, None] - score[None, :])[mask]).mean()


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(frame):
        return frame.reset_index(drop=True)
    ranks = frame["D"].rank(method="first", pct=True)
    bins = np.minimum((ranks.to_numpy() * 10).astype(int), 9)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for group in range(10):
        candidates = np.flatnonzero(bins == group)
        take = min(len(candidates), max(1, round(n / 10)))
        chosen.append(rng.choice(candidates, size=take, replace=False))
    indices = np.concatenate(chosen)
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    elif len(indices) < n:
        remaining = np.setdiff1d(np.arange(len(frame)), indices, assume_unique=False)
        indices = np.concatenate([indices, rng.choice(remaining, size=n - len(indices), replace=False)])
    return frame.iloc[np.sort(indices)].reset_index(drop=True)


def ndcg(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(len(y) * fraction)))
    ideal = np.sort(y)[::-1][:k]
    selected = y[np.argsort(score)[::-1][:k]]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.dot(selected, discount) / max(np.dot(ideal, discount), 1e-12))


def top_overlap(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(len(y) * fraction)))
    return float(len(set(np.argsort(y)[-k:]) & set(np.argsort(score)[-k:])) / k)


def evaluate(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    pair_n = min(500_000, len(y) * (len(y) - 1) // 2)
    left = rng.integers(0, len(y), size=pair_n)
    right = rng.integers(0, len(y), size=pair_n)
    valid = y[left] != y[right]
    pairwise = float(np.mean((score[left][valid] - score[right][valid]) * (y[left][valid] - y[right][valid]) > 0)) if valid.any() else 1.0
    calibrated = LinearRegression().fit(score.reshape(-1, 1), y).predict(score.reshape(-1, 1))
    return {
        "spearman": float(spearmanr(y, score).statistic),
        "kendall_tau": float(kendalltau(y, score).statistic),
        "pairwise_accuracy": pairwise,
        "ndcg_at_05pct": ndcg(y, score, 0.05),
        "ndcg_at_10pct": ndcg(y, score, 0.10),
        "ndcg_at_20pct": ndcg(y, score, 0.20),
        "ef_at_05pct": top_overlap(y, score, 0.05) / 0.05,
        "ef_at_10pct": top_overlap(y, score, 0.10) / 0.10,
        "ef_at_20pct": top_overlap(y, score, 0.20) / 0.20,
        "mae_calibrated": float(mean_absolute_error(y, calibrated)),
        "pcc": float(pearsonr(y, calibrated).statistic),
        "r2": float(r2_score(y, calibrated)),
        "rmse": float(mean_squared_error(y, calibrated) ** 0.5),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for raw in loader:
        batch = to_device(raw, device)
        mu, rank = model(batch)
        magnitude = torch.linalg.vector_norm(mu, dim=1)
        for idx, domain_index in enumerate(raw["domain_index"].numpy()):
            rows.append({
                "domain_index": int(domain_index),
                "D": float(raw["target_d"][idx]),
                "pred__vector_mu": float(magnitude[idx].cpu()),
                "pred__vector_rank": float(rank[idx].cpu()),
                "pred_mu_x": float(mu[idx, 0].cpu()),
                "pred_mu_y": float(mu[idx, 1].cpu()),
                "pred_mu_z": float(mu[idx, 2].cpu()),
            })
    return pd.DataFrame(rows)


def load_meta() -> pd.DataFrame:
    manifest = json.loads(ALIGN_MANIFEST.read_text(encoding="utf-8"))
    if float(manifest["coverage"]) < 0.99 or float(manifest["max_target_D_reconstruction_error"]) > 1e-5:
        raise RuntimeError("Vector alignment manifest did not pass required gates")
    base = pd.read_csv(CV, low_memory=False)
    excluded = set(pd.read_csv(EXCLUSION, usecols=["domain_index"])["domain_index"].astype(int))
    base = base.loc[~base["domain_index"].astype(int).isin(excluded)].reset_index(drop=True)
    aligned = pd.read_pickle(ALIGN)
    frame = base.merge(aligned.drop(columns=["D", "cv_fold"], errors="ignore").drop_duplicates("domain_index"), on="domain_index", how="inner", validate="many_to_one")
    frame = frame.loc[frame["status"].eq("ok")].copy().reset_index(drop=True)
    if len(frame) < int(60672 * 0.99):
        raise RuntimeError(f"Aligned complete-case coverage fell below 99%: {len(frame)}/60672")
    if not np.allclose(frame["D"], frame["target_D"], atol=1e-5):
        raise RuntimeError("Aligned target vector norm does not reproduce D")
    return frame


def refit_all_development(args: argparse.Namespace) -> None:
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    frame = load_meta()
    data = VectorDataset(frame, frame["D"].to_numpy(float))
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0,
    )
    model = XTBVectorDipole(hidden=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    y_std = float(frame["D"].std())
    out = RUN / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "full_history.csv"
    resume_path = out / "full_resume.pt"
    history = []
    start_epoch = 1
    if args.resume_full:
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-full requested but no checkpoint exists in {out}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        start_epoch = int(checkpoint.get("last_epoch", checkpoint.get("epoch", 0))) + 1
        print(f"resuming full vector from epoch={start_epoch}", flush=True)
    total_epochs = int(args.high_epochs + args.low_epochs)
    for epoch in range(start_epoch, total_epochs + 1):
        lr = float(args.learning_rate if epoch <= args.high_epochs else args.low_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = 0.0
        started = time.perf_counter()
        for raw in loader:
            batch = to_device(raw, device)
            if args.rotate_augment:
                rotation = random_rotation(device, batch["pos"].dtype)
                batch["pos"] = batch["pos"] @ rotation
                batch["baseline_mu"] = batch["baseline_mu"] @ rotation
                batch["target_mu"] = batch["target_mu"] @ rotation
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                mu, rank = model(batch)
                magnitude = torch.linalg.vector_norm(mu, dim=1)
                vector_loss = F.smooth_l1_loss(mu, batch["target_mu"])
                magnitude_loss = F.smooth_l1_loss((magnitude - batch["target_d"]) / y_std, torch.zeros_like(magnitude))
                rank_loss = F.smooth_l1_loss(rank, batch["rank_target"])
                loss = 0.50 * vector_loss + 0.25 * magnitude_loss + 0.15 * rank_loss + 0.10 * pairwise_loss(magnitude, batch["target_d"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu())
        history.append({"epoch": epoch, "learning_rate": lr, "train_loss": total / max(len(loader), 1),
                        "elapsed_sec": time.perf_counter() - started})
        pd.DataFrame(history).to_csv(history_path, index=False)
        torch.save({
            "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "last_epoch": epoch, "args": vars(args),
        }, resume_path)
        print(f"full vector epoch={epoch} lr={lr:.2e} loss={history[-1]['train_loss']:.5f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "epoch": total_epochs, "args": vars(args)}, out / "full_vector.pt")
    manifest = {
        "status": "frozen_fold0_schedule_refit_on_all_development", "n_candidate100_clean_graph_valid": len(frame),
        "schedule": {"high_epochs": int(args.high_epochs), "high_learning_rate": float(args.learning_rate),
                     "low_epochs": int(args.low_epochs), "low_learning_rate": float(args.low_learning_rate)},
        "device": str(device), "locked_status": "not read",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_fold(args: argparse.Namespace) -> None:
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED + args.fold)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    frame = load_meta()
    fold = int(args.fold)
    outer_train = frame.loc[frame["cv_fold"].ne(fold)].copy().reset_index(drop=True)
    outer_val = frame.loc[frame["cv_fold"].eq(fold)].copy().reset_index(drop=True)
    outer_train = stratified_sample(outer_train, args.train_samples, SEED + fold)
    outer_val = stratified_sample(outer_val, args.val_samples, SEED + fold + 100)
    inner_mask = ((outer_train["domain_index"].astype(int) + SEED + fold) % 10 == 0).to_numpy()
    fit = outer_train.loc[~inner_mask].reset_index(drop=True)
    inner = outer_train.loc[inner_mask].reset_index(drop=True)
    rank_reference = fit["D"].to_numpy(float)
    train_data = VectorDataset(fit, rank_reference)
    inner_data = VectorDataset(inner, inner["D"].to_numpy(float))
    outer_data = VectorDataset(outer_val, outer_val["D"].to_numpy(float))
    sampler = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0)
    inner_loader = DataLoader(inner_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0)
    outer_loader = DataLoader(outer_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0)
    model = XTBVectorDipole(hidden=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    y_std = float(fit["D"].std())
    out = RUN / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / f"fold{fold}_best.pt"
    resume_path = out / f"fold{fold}_resume.pt"
    history_path = out / f"fold{fold}_history.csv"
    epoch_metrics_path = out / f"fold{fold}_epoch_metrics.csv"
    if args.evaluate_only_checkpoint:
        source = Path(args.evaluate_only_checkpoint)
        if not source.exists():
            raise FileNotFoundError(f"--evaluate-only-checkpoint does not exist: {source}")
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        predicted = predict(model, outer_loader, device)
        if predicted["domain_index"].tolist() != outer_val["domain_index"].tolist():
            raise RuntimeError("Outer vector predictions are not in the fixed validation-record order")
        pred = outer_val[["domain_index", "canonical_smiles", "D", "cv_fold"]].copy()
        for column in predicted.columns:
            if column not in {"domain_index", "D"}:
                pred[column] = predicted[column].to_numpy()
        pred.to_csv(out / f"fold{fold}_predictions.csv", index=False)
        metrics = [{"fold": fold, "member": column, **evaluate(pred["D"].to_numpy(float), pred[column].to_numpy(float))}
                   for column in ("pred__vector_mu", "pred__vector_rank")]
        pd.DataFrame(metrics).to_csv(out / f"fold{fold}_metrics.csv", index=False)
        manifest = {
            "run_name": args.run_name, "fold": fold, "n_complete_case": len(frame), "n_fit": len(fit),
            "n_inner": len(inner), "n_outer": len(outer_val), "device": str(device),
            "evaluation_only_checkpoint": str(source), "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_inner_score": float(checkpoint.get("inner_score", float("nan"))),
            "locked_status": "not read", "metrics": metrics,
        }
        (out / f"fold{fold}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(pd.DataFrame(metrics).to_string(index=False), flush=True)
        return
    history = []
    epoch_metrics: list[dict] = []
    best, stale, start_epoch = -np.inf, 0, 1
    if args.resume:
        restore_path = resume_path if resume_path.exists() else checkpoint_path
        if not restore_path.exists():
            raise FileNotFoundError(f"--resume requested but no checkpoint exists in {out}")
        restore = torch.load(restore_path, map_location=device, weights_only=False)
        model.load_state_dict(restore["state_dict"])
        if "optimizer_state_dict" in restore:
            optimizer.load_state_dict(restore["optimizer_state_dict"])
        if "scaler_state_dict" in restore:
            scaler.load_state_dict(restore["scaler_state_dict"])
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        if epoch_metrics_path.exists():
            epoch_metrics = pd.read_csv(epoch_metrics_path).to_dict("records")
        best = float(restore.get("best_score", restore.get("inner_score", -np.inf)))
        stale = int(restore.get("stale", 0))
        start_epoch = int(restore.get("last_epoch", restore.get("epoch", 0))) + 1
        # Backfill a complete metric snapshot for the checkpoint that was
        # saved before per-epoch metric persistence was introduced.
        if not epoch_metrics:
            checkpoint_epoch = int(restore.get("epoch", start_epoch - 1))
            checkpoint_pred = predict(model, inner_loader, device)
            for member in ("pred__vector_mu", "pred__vector_rank"):
                epoch_metrics.append({"epoch": checkpoint_epoch, "member": member, **evaluate(checkpoint_pred["D"].to_numpy(float), checkpoint_pred[member].to_numpy(float))})
            pd.DataFrame(epoch_metrics).to_csv(epoch_metrics_path, index=False)
        print(f"resuming fold={fold} from epoch={start_epoch} with best={best:.5f}, stale={stale}", flush=True)
    elif args.init_checkpoint:
        source = Path(args.init_checkpoint)
        if not source.exists():
            raise FileNotFoundError(f"--init-checkpoint does not exist: {source}")
        initial = torch.load(source, map_location=device, weights_only=False)
        model.load_state_dict(initial["state_dict"])
        print(
            f"initializing fold={fold} from checkpoint={source} "
            f"(source_epoch={initial.get('epoch', 'unknown')}) with a fresh optimizer",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        started = time.perf_counter()
        for raw in train_loader:
            batch = to_device(raw, device)
            if args.rotate_augment:
                rotation = random_rotation(device, batch["pos"].dtype)
                batch["pos"] = batch["pos"] @ rotation
                batch["baseline_mu"] = batch["baseline_mu"] @ rotation
                batch["target_mu"] = batch["target_mu"] @ rotation
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                mu, rank = model(batch)
                magnitude = torch.linalg.vector_norm(mu, dim=1)
                vector_loss = F.smooth_l1_loss(mu, batch["target_mu"])
                magnitude_loss = F.smooth_l1_loss((magnitude - batch["target_d"]) / y_std, torch.zeros_like(magnitude))
                rank_loss = F.smooth_l1_loss(rank, batch["rank_target"])
                loss = 0.50 * vector_loss + 0.25 * magnitude_loss + 0.15 * rank_loss + 0.10 * pairwise_loss(magnitude, batch["target_d"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu())
        inner_pred = predict(model, inner_loader, device)
        inner_mu_sp = float(spearmanr(inner_pred["D"], inner_pred["pred__vector_mu"]).statistic)
        inner_rank_sp = float(spearmanr(inner_pred["D"], inner_pred["pred__vector_rank"]).statistic)
        for member in ("pred__vector_mu", "pred__vector_rank"):
            epoch_metrics.append({"epoch": epoch, "member": member, **evaluate(inner_pred["D"].to_numpy(float), inner_pred[member].to_numpy(float))})
        pd.DataFrame(epoch_metrics).to_csv(epoch_metrics_path, index=False)
        score = 0.75 * inner_mu_sp + 0.25 * inner_rank_sp
        history.append({"epoch": epoch, "loss": total / max(len(train_loader), 1), "inner_mu_spearman": inner_mu_sp, "inner_rank_spearman": inner_rank_sp, "selection_score": score, "elapsed_sec": time.perf_counter() - started})
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"vector fold={fold} epoch={epoch} loss={history[-1]['loss']:.5f} inner_mu={inner_mu_sp:.5f} inner_rank={inner_rank_sp:.5f}", flush=True)
        if score > best + 1e-4:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "args": vars(args), "inner_score": score}, checkpoint_path)
        else:
            stale += 1
        # This is written each epoch, unlike the best checkpoint, so an
        # interruption can resume optimizer and AMP state without replaying.
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "last_epoch": epoch,
            "best_score": best,
            "stale": stale,
            "args": vars(args),
        }, resume_path)
        if stale >= args.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    predicted = predict(model, outer_loader, device)
    if predicted["domain_index"].tolist() != outer_val["domain_index"].tolist():
        raise RuntimeError("Outer vector predictions are not in the fixed validation-record order")
    pred = outer_val[["domain_index", "canonical_smiles", "D", "cv_fold"]].copy()
    for column in predicted.columns:
        if column not in {"domain_index", "D"}:
            pred[column] = predicted[column].to_numpy()
    pred.to_csv(out / f"fold{fold}_predictions.csv", index=False)
    metrics = []
    for column in ("pred__vector_mu", "pred__vector_rank"):
        metrics.append({"fold": fold, "member": column, **evaluate(pred["D"].to_numpy(float), pred[column].to_numpy(float))})
    pd.DataFrame(metrics).to_csv(out / f"fold{fold}_metrics.csv", index=False)
    manifest = {
        "run_name": args.run_name, "fold": fold, "n_complete_case": len(frame), "n_fit": len(fit), "n_inner": len(inner), "n_outer": len(outer_val),
        "device": str(device), "best_epoch": int(checkpoint["epoch"]), "best_inner_score": float(checkpoint["inner_score"]),
        "architecture": "xTB baseline vector + zero-sum charge correction + local equivariant atomic dipoles",
        "inputs": "xTB three-conformer coordinates/charges/energy/full dipole plus graph atom attributes",
        "reference_geometry_at_inference": False, "locked_status": "not read",
        "initial_checkpoint": args.init_checkpoint or None,
        "metrics": metrics,
    }
    (out / f"fold{fold}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-name", default="domain65k_d_vector_equivariant")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-samples", type=int, default=0, help="Optional stratified outer-training subset for a fast diagnostic.")
    parser.add_argument("--val-samples", type=int, default=0, help="Optional stratified outer-validation subset for a fast diagnostic.")
    parser.add_argument("--rotate-augment", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume the latest fold checkpoint and history in the run directory.")
    parser.add_argument("--init-checkpoint", default=None, help="Initialize a new run from model weights while resetting optimizer, history, and early stopping.")
    parser.add_argument("--evaluate-only-checkpoint", default=None, help="Evaluate a frozen checkpoint on the outer fold without training.")
    parser.add_argument("--all-development", action="store_true", help="Refit a fixed training schedule on all candidate100-clean graph-valid development rows.")
    parser.add_argument("--high-epochs", type=int, default=35)
    parser.add_argument("--low-epochs", type=int, default=1)
    parser.add_argument("--low-learning-rate", type=float, default=5e-5)
    parser.add_argument("--resume-full", action="store_true", help="Resume an interrupted all-development fixed schedule.")
    args = parser.parse_args()
    if args.all_development:
        refit_all_development(args)
    else:
        run_fold(args)


if __name__ == "__main__":
    main()
