from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from skin_lesion_risk.evaluation.calibration import expected_calibration_error
from skin_lesion_risk.evaluation.metrics import binary_classification_metrics, subgroup_metric_gaps
from skin_lesion_risk.evaluation.thresholds import threshold_at_min_sensitivity, youden_threshold
from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult


class RelationSAGEBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float, edge_chunk_size: int = 250_000) -> None:
        super().__init__()
        self.relation_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)])
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_chunk_size = edge_chunk_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            out = self.self_linear(x)
            return self.norm(F.gelu(out))
        agg = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        src_all, dst_all = edge_index[0], edge_index[1]
        for rel, linear in enumerate(self.relation_linears):
            mask = edge_type == rel
            if not torch.any(mask):
                continue
            transformed = linear(x)
            src_rel = src_all[mask]
            dst_rel = dst_all[mask]
            weight_rel = edge_weight[mask].to(x.dtype)
            for start in range(0, src_rel.numel(), self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, src_rel.numel())
                src = src_rel[start:end]
                dst = dst_rel[start:end]
                weight = weight_rel[start:end].unsqueeze(1)
                msg = transformed[src] * weight
                agg.index_add_(0, dst, msg)
                degree.index_add_(0, dst, weight.clamp_min(1e-6))
        agg = agg / degree.clamp_min(1.0)
        out = self.self_linear(x) + agg
        out = self.dropout(F.gelu(out))
        return self.norm(x + out)


class PrototypeAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, initial: torch.Tensor, graph: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prototypes.numel() == 0:
            empty = torch.zeros_like(graph)
            weights = torch.empty((graph.shape[0], 0), device=graph.device)
            return empty, weights
        q = self.query(torch.cat([initial, graph], dim=1))
        scores = q @ prototypes.T / max(float(q.shape[1]) ** 0.5, 1.0)
        weights = torch.softmax(scores, dim=1)
        return weights @ prototypes, weights


class LGKEGNN(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_relations: int,
        dropout: float,
        edge_chunk_size: int = 250_000,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList(
            [RelationSAGEBlock(hidden_dim, num_relations, dropout, edge_chunk_size=edge_chunk_size) for _ in range(num_layers)]
        )
        self.prototype_attention = PrototypeAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = graph["x"]
        edge_index = graph["edge_index"]
        edge_type = graph["edge_type"]
        edge_weight = graph["edge_weight"]
        lesion_indices = graph["lesion_indices"]
        prototype_indices = prototype_node_indices(graph)

        h0 = self.input_proj(x)
        h = h0
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_weight)

        lesion_h0 = h0[lesion_indices]
        lesion_h = h[lesion_indices]
        prototypes = h[prototype_indices] if len(prototype_indices) else h.new_empty((0, h.shape[1]))
        knowledge, attention = self.prototype_attention(lesion_h0, lesion_h, prototypes)
        logits = self.head(torch.cat([lesion_h0, lesion_h, knowledge], dim=1)).squeeze(1)
        return {"logits": logits, "probs": torch.sigmoid(logits), "proto_attention": attention, "node_embeddings": lesion_h}


class LGKEGNNModelAdapter(BaseModelAdapter):
    """Native PyTorch LGKE-GNN adapter for lesion graph risk classification."""

    model_type = "graph_multimodal"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, params=params)
        self.model: LGKEGNN | None = None
        self.device = torch.device(str(self.params.get("device", "cpu")))
        self.logs: list[dict[str, float]] = []
        self.best_thresholds: dict[str, float] = {}
        self.best_state: dict[str, torch.Tensor] | None = None

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "LGKEGNNModelAdapter":
        if train.graph is None:
            raise ValueError("LGKEGNNModelAdapter.fit requires ModelBatch.graph.")
        train_graph = move_graph(train.graph, self.device)
        valid_graph = move_graph(valid.graph, self.device) if valid and valid.graph is not None else None
        input_dim = int(train_graph.get("input_dim", train_graph["x"].shape[1]))
        num_relations = len(train_graph.get("relation_names", ())) or int(train_graph["edge_type"].max().item() + 1)
        hidden_dim = int(self.params.get("hidden_dim", 256))
        num_layers = int(self.params.get("num_layers", self.params.get("gnn_layers", 2)))
        dropout = float(self.params.get("dropout", 0.2))
        edge_chunk_size = int(self.params.get("edge_chunk_size", 250_000))
        epochs = int(self.params.get("epochs", 50))
        patience = int(self.params.get("patience", 8))
        lr = float(self.params.get("learning_rate", 3e-4))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        out_dir = Path(str(self.params.get("out_dir", "data/artifacts/trained_models/m5_lgke_gnn/fold0"))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model = LGKEGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_relations=num_relations,
            dropout=dropout,
            edge_chunk_size=edge_chunk_size,
        ).to(self.device)
        train_labels = train_graph["y"][train_graph["train_mask"]]
        positives = float(train_labels.sum().item())
        negatives = float(len(train_labels) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        initial_checkpoint = self.params.get("initial_checkpoint")
        if initial_checkpoint:
            self._load_initial_state(initial_checkpoint)

        best_metric = -float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(train_graph)
            train_idx = train_graph["train_mask"].nonzero(as_tuple=False).squeeze(1)
            logits = out["logits"][train_idx]
            labels = train_graph["y"][train_idx]
            loss = self._train_loss(logits, labels, train_graph, train_idx, pos_weight)
            smooth = graph_smoothing_loss(out["node_embeddings"], train_graph, lambda_value=float(self.params.get("lambda_smooth", 0.0)))
            total_loss = loss + smooth
            total_loss.backward()
            optimizer.step()

            row = {"epoch": float(epoch), "train_loss": float(total_loss.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"])}
            if valid_graph is not None:
                val_result = self._predict_graph(valid_graph, eval_only=True)
                val_labels = val_result.labels if val_result.labels is not None else np.array([])
                val_scores = val_result.scores
                val_threshold = youden_threshold(val_labels, val_scores) if len(np.unique(val_labels)) == 2 else 0.5
                high_threshold = threshold_at_min_sensitivity(val_labels, val_scores, min_sensitivity=0.90) if len(np.unique(val_labels)) == 2 else 0.5
                val_metrics = binary_classification_metrics(val_labels, val_scores, threshold=val_threshold)
                row.update(
                    {
                        "val_loss": float(self._eval_loss(valid_graph, criterion)),
                        "val_pauc_tpr80": float(val_metrics.get("pauc_tpr80", float("nan"))),
                        "val_auprc": float(val_metrics.get("auprc", float("nan"))),
                        "val_auroc": float(val_metrics.get("auroc", float("nan"))),
                        "val_brier": float(val_metrics.get("brier", float("nan"))),
                        "val_ece": float(expected_calibration_error(val_labels, val_scores)),
                        "threshold_youden": float(val_threshold),
                        "threshold_sensitivity_0_90": float(high_threshold),
                    }
                )
                metric = row["val_pauc_tpr80"] if np.isfinite(row["val_pauc_tpr80"]) else row.get("val_auprc", -float("inf"))
            else:
                metric = -float(total_loss.detach().cpu())

            self.logs.append(row)
            append_log(out_dir / "train_log.csv", row)
            append_log(out_dir / "loss_curve.csv", {"epoch": row["epoch"], "train_loss": row["train_loss"], "val_loss": row.get("val_loss", float("nan"))})

            if metric > best_metric:
                best_metric = float(metric)
                bad_epochs = 0
                self.best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                self.best_thresholds = {
                    "youden": float(row.get("threshold_youden", 0.5)),
                    "sensitivity_at_least_0_90": float(row.get("threshold_sensitivity_0_90", 0.5)),
                }
                self.save(out_dir / "best.ckpt")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        if batch.graph is None:
            raise ValueError("LGKEGNNModelAdapter.predict_proba requires ModelBatch.graph.")
        return self._predict_graph(move_graph(batch.graph, self.device), eval_only=True)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save before model is initialized.")
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "state_dict": self.best_state or {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "model_config": {
                "input_dim": int(self.model.input_proj[0].in_features),
                "hidden_dim": int(self.model.input_proj[0].out_features),
                "num_layers": len(self.model.layers),
                "num_relations": len(self.model.layers[0].relation_linears) if self.model.layers else 0,
                "dropout": float(self.params.get("dropout", 0.2)),
                "edge_chunk_size": int(self.params.get("edge_chunk_size", 250_000)),
            },
            "thresholds": self.best_thresholds,
            "logs": self.logs,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "LGKEGNNModelAdapter":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(model_name=payload.get("model_name", "m5_lgke_gnn"), params=payload.get("params", {}))
        model_config = payload.get("model_config")
        if model_config:
            adapter.model = LGKEGNN(
                input_dim=int(model_config["input_dim"]),
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                num_relations=int(model_config["num_relations"]),
                dropout=float(model_config.get("dropout", adapter.params.get("dropout", 0.2))),
                edge_chunk_size=int(model_config.get("edge_chunk_size", adapter.params.get("edge_chunk_size", 250_000))),
            )
            adapter.model.load_state_dict(payload["state_dict"])
        adapter.best_state = payload["state_dict"]
        adapter.best_thresholds = payload.get("thresholds", {})
        adapter.logs = payload.get("logs", [])
        adapter.is_fitted = True
        return adapter

    def _predict_graph(self, graph: dict[str, Any], *, eval_only: bool) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph)
        indices = graph["eval_indices"] if eval_only else graph["lesion_indices"]
        scores = out["probs"][indices].detach().cpu().numpy()
        labels = graph["y"][indices].detach().cpu().numpy().astype(int)
        sample_ids = [graph["sample_ids"][int(i)] for i in indices.detach().cpu().numpy().tolist()]
        return PredictionResult(sample_ids=sample_ids, scores=scores, labels=labels)

    def _eval_loss(self, graph: dict[str, Any], criterion: nn.Module) -> float:
        if self.model is None:
            return float("nan")
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph)
            idx = graph["eval_indices"]
            loss = criterion(out["logits"][idx], graph["y"][idx])
        return float(loss.detach().cpu())

    def _train_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        graph: dict[str, Any],
        train_idx: torch.Tensor,
        pos_weight: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
        source_weights = self.params.get("source_loss_weights")
        if not source_weights:
            return loss.mean()
        sources = graph.get("source")
        if not sources:
            return loss.mean()
        weight_values = []
        for idx in train_idx.detach().cpu().numpy().tolist():
            source = str(sources[int(idx)])
            weight_values.append(float(source_weights.get(source, source_weights.get("*", 1.0))))
        weights = torch.tensor(weight_values, dtype=loss.dtype, device=loss.device)
        return torch.sum(loss * weights) / weights.sum().clamp_min(1e-6)

    def _load_initial_state(self, checkpoint: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot load initial state before model is initialized.")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[LGKEGNN] loaded initial checkpoint with missing={len(missing)} unexpected={len(unexpected)}: {checkpoint}",
                flush=True,
            )


def move_graph(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in graph.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def prototype_node_indices(graph: dict[str, Any]) -> torch.Tensor:
    names = graph.get("node_type", [])
    indices = [idx for idx, value in enumerate(names) if "prototype" in str(value)]
    device = graph["x"].device
    return torch.tensor(indices, dtype=torch.long, device=device)


def graph_smoothing_loss(node_embeddings: torch.Tensor, graph: dict[str, Any], *, lambda_value: float) -> torch.Tensor:
    if lambda_value <= 0 or graph["edge_index"].numel() == 0:
        return node_embeddings.new_tensor(0.0)
    relation_names = list(graph.get("relation_names", ()))
    smooth_relations = {idx for idx, name in enumerate(relation_names) if name in {"visual_knn", "metadata_knn"}}
    if not smooth_relations:
        return node_embeddings.new_tensor(0.0)
    edge_type = graph["edge_type"]
    mask = torch.zeros_like(edge_type, dtype=torch.bool)
    for rel in smooth_relations:
        mask |= edge_type == rel
    src, dst = graph["edge_index"][:, mask]
    lesion_count = len(graph["lesion_indices"])
    lesion_mask = (src < lesion_count) & (dst < lesion_count)
    if not torch.any(lesion_mask):
        return node_embeddings.new_tensor(0.0)
    src = src[lesion_mask]
    dst = dst[lesion_mask]
    weights = graph["edge_weight"][mask][lesion_mask].to(node_embeddings.dtype)
    diffs = node_embeddings[src] - node_embeddings[dst]
    return lambda_value * torch.mean(weights * torch.sum(diffs * diffs, dim=1))


def append_log(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def prediction_metrics(result: PredictionResult, *, threshold: float, groups: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    if result.labels is None:
        return {}
    values = binary_classification_metrics(result.labels, result.scores, threshold=threshold)
    values["ece"] = expected_calibration_error(result.labels, result.scores)
    output: dict[str, Any] = {"values": values, "threshold": threshold}
    if groups:
        output["subgroup_gaps"] = subgroup_metric_gaps(result.labels, result.scores, groups, threshold=threshold)
    return output


def write_predictions(path: str | Path, result: PredictionResult) -> None:
    rows = result.to_records()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "score", "label"])
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
