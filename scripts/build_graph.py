from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_lesion_risk.data.preprocessing import FoldTabularPreprocessor


RELATION_NAMES = (
    "same_patient",
    "has_attribute",
    "visual_knn",
    "metadata_knn",
    "visual_prototype",
    "metadata_prototype",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build native PyTorch graph files for LGKE-GNN.")
    parser.add_argument("--manifest", default="data/processed/manifest_isic.csv")
    parser.add_argument("--folds", default="data/processed/folds_isic.csv")
    parser.add_argument("--preprocessor-dir", default="data/processed/preprocessors")
    parser.add_argument("--out-dir", default="data/processed/graph")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k-visual", type=int, default=10)
    parser.add_argument("--k-metadata", type=int, default=10)
    parser.add_argument("--n-visual-prototypes", type=int, default=64)
    parser.add_argument("--n-metadata-prototypes", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-feature-cache", default="data/processed/embeddings/image_stats_features.npz")
    parser.add_argument("--refresh-image-cache", action="store_true")
    parser.add_argument("--knn-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--knn-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional per-graph cap for CPU smoke tests.")
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / args.manifest)
    folds = pd.read_csv(ROOT / args.folds)
    preprocessor = FoldTabularPreprocessor.load(ROOT / args.preprocessor_dir / f"fold{args.fold}_tabular.pkl")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_ids = {
        split: folds[(folds["fold"] == args.fold) & (folds["split"] == split)]["sample_id"].astype(str).tolist()
        for split in ("train", "val", "test")
    }
    if args.max_samples:
        split_ids = cap_split_ids(manifest, split_ids, args.max_samples, seed=args.seed)

    train_df = manifest[manifest["sample_id"].astype(str).isin(split_ids["train"])].copy()
    active_ids = sorted(set().union(*(set(v) for v in split_ids.values())))
    image_cache = load_or_build_image_feature_cache(
        manifest[manifest["sample_id"].astype(str).isin(active_ids)].copy() if args.max_samples else manifest,
        ROOT / args.image_feature_cache,
        image_size=args.image_size,
        refresh=args.refresh_image_cache,
    )
    knn_device = resolve_knn_device(args.knn_device)
    reports: dict[str, object] = {"fold": args.fold, "splits": {}, "relations": RELATION_NAMES}
    for split in ("train", "val", "test"):
        query_df = train_df if split == "train" else manifest[manifest["sample_id"].astype(str).isin(split_ids[split])].copy()
        graph_df = train_df if split == "train" else pd.concat([train_df, query_df], ignore_index=True)
        graph = build_graph(
            graph_df=graph_df,
            train_df=train_df,
            query_ids=set(query_df["sample_id"].astype(str)),
            split=split,
            preprocessor=preprocessor,
            image_cache=image_cache,
            k_visual=args.k_visual,
            k_metadata=args.k_metadata,
            n_visual_prototypes=args.n_visual_prototypes,
            n_metadata_prototypes=args.n_metadata_prototypes,
            image_size=args.image_size,
            knn_device=knn_device,
            knn_chunk_size=args.knn_chunk_size,
            seed=args.seed,
        )
        path = out_dir / f"fold{args.fold}_{split}_graph.pt"
        torch.save(graph, path)
        edge_counts = relation_counts(graph["edge_type"].numpy())
        reports["splits"][split] = {
            "path": str(path),
            "n_nodes": int(graph["x"].shape[0]),
            "n_lesions": int(len(graph["lesion_indices"])),
            "n_eval_lesions": int(len(graph["eval_indices"])),
            "n_edges": int(graph["edge_index"].shape[1]),
            "edge_counts": edge_counts,
        }
        write_nodes_edges_tables(graph, out_dir, args.fold, split)

    (out_dir / f"fold{args.fold}_graph_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


def build_graph(
    *,
    graph_df: pd.DataFrame,
    train_df: pd.DataFrame,
    query_ids: set[str],
    split: str,
    preprocessor: FoldTabularPreprocessor,
    image_cache: dict[str, np.ndarray],
    k_visual: int,
    k_metadata: int,
    n_visual_prototypes: int,
    n_metadata_prototypes: int,
    image_size: int,
    knn_device: str,
    knn_chunk_size: int,
    seed: int,
) -> dict[str, object]:
    graph_df = preprocessor.add_bins(graph_df.reset_index(drop=True))
    train_df = preprocessor.add_bins(train_df.reset_index(drop=True))
    sample_ids = graph_df["sample_id"].astype(str).tolist()
    train_sample_ids = set(train_df["sample_id"].astype(str).tolist())
    lesion_count = len(graph_df)

    meta = preprocessor.transform(graph_df).to_numpy(dtype=np.float32)
    image = np.vstack([image_feature(row, image_cache, image_size=image_size) for _, row in graph_df.iterrows()]).astype(np.float32)
    lesion_x = np.concatenate([image, meta], axis=1)
    feature_dim = lesion_x.shape[1]
    node_features: list[np.ndarray] = [lesion_x]
    node_type: list[str] = ["lesion"] * lesion_count
    node_name: list[str] = sample_ids.copy()

    patient_offset = lesion_count
    patient_nodes, patient_map = make_patient_nodes(graph_df, lesion_x)
    node_features.append(patient_nodes)
    node_type.extend(["patient"] * len(patient_nodes))
    node_name.extend([f"patient={p}" for p in patient_map])

    attr_offset = patient_offset + len(patient_nodes)
    attr_nodes, attr_map = make_attribute_nodes(graph_df, train_df, lesion_x, feature_dim)
    node_features.append(attr_nodes)
    node_type.extend(["attribute"] * len(attr_nodes))
    node_name.extend(attr_map)

    visual_proto_offset = attr_offset + len(attr_nodes)
    visual_proto = make_kmeans_prototypes(image[: len(train_df)], n_visual_prototypes, seed=seed, feature_dim=image.shape[1])
    visual_proto_x = np.zeros((len(visual_proto), feature_dim), dtype=np.float32)
    visual_proto_x[:, : image.shape[1]] = visual_proto
    node_features.append(visual_proto_x)
    node_type.extend(["visual_prototype"] * len(visual_proto_x))
    node_name.extend([f"visual_proto={i}" for i in range(len(visual_proto_x))])

    metadata_proto_offset = visual_proto_offset + len(visual_proto_x)
    metadata_proto = make_kmeans_prototypes(meta[: len(train_df)], n_metadata_prototypes, seed=seed + 13, feature_dim=meta.shape[1])
    metadata_proto_x = np.zeros((len(metadata_proto), feature_dim), dtype=np.float32)
    metadata_proto_x[:, image.shape[1] :] = metadata_proto
    node_features.append(metadata_proto_x)
    node_type.extend(["metadata_prototype"] * len(metadata_proto_x))
    node_name.extend([f"metadata_proto={i}" for i in range(len(metadata_proto_x))])

    x = np.vstack(node_features).astype(np.float32)
    edges: list[tuple[int, int, int, float]] = []
    edges.extend(patient_edges(graph_df, patient_map, patient_offset))
    edges.extend(attribute_edges(graph_df, attr_map, attr_offset))
    edges.extend(
        knn_edges(
            image,
            train_size=len(train_df),
            k=k_visual,
            relation=RELATION_NAMES.index("visual_knn"),
            split=split,
            device=knn_device,
            chunk_size=knn_chunk_size,
        )
    )
    edges.extend(
        knn_edges(
            meta,
            train_size=len(train_df),
            k=k_metadata,
            relation=RELATION_NAMES.index("metadata_knn"),
            split=split,
            device=knn_device,
            chunk_size=knn_chunk_size,
        )
    )
    edges.extend(prototype_edges(image, visual_proto, visual_proto_offset, RELATION_NAMES.index("visual_prototype"), topk=3))
    edges.extend(prototype_edges(meta, metadata_proto, metadata_proto_offset, RELATION_NAMES.index("metadata_prototype"), topk=3))

    if edges:
        edge_array = np.asarray(edges, dtype=np.float32)
        edge_index = torch.tensor(edge_array[:, :2].T, dtype=torch.long)
        edge_type = torch.tensor(edge_array[:, 2], dtype=torch.long)
        edge_weight = torch.tensor(edge_array[:, 3], dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    labels = torch.tensor(graph_df["target"].astype(int).to_numpy(), dtype=torch.float32)
    train_mask = torch.tensor([sid in train_sample_ids for sid in sample_ids], dtype=torch.bool)
    eval_mask = torch.tensor([sid in query_ids for sid in sample_ids], dtype=torch.bool)
    if split == "train":
        eval_mask = train_mask.clone()
    lesion_indices = torch.arange(lesion_count, dtype=torch.long)
    eval_indices = lesion_indices[eval_mask]

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_weight": edge_weight,
        "y": labels,
        "train_mask": train_mask,
        "eval_mask": eval_mask,
        "lesion_indices": lesion_indices,
        "eval_indices": eval_indices,
        "sample_ids": sample_ids,
        "source": graph_df["source"].fillna("unknown").astype(str).tolist() if "source" in graph_df.columns else ["unknown"] * len(sample_ids),
        "node_type": node_type,
        "node_name": node_name,
        "relation_names": RELATION_NAMES,
        "split": split,
        "input_dim": int(feature_dim),
    }


def image_feature(row: pd.Series, cache: dict[str, np.ndarray], *, image_size: int) -> np.ndarray:
    sample_id = str(row["sample_id"])
    if sample_id in cache:
        return cache[sample_id]
    path = Path(str(row["image_path"]))
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size))
            stat = ImageStat.Stat(img)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            mean = np.asarray(stat.mean, dtype=np.float32) / 255.0
            std = np.asarray(stat.stddev, dtype=np.float32) / 255.0
            q25 = np.quantile(arr.reshape(-1, 3), 0.25, axis=0).astype(np.float32)
            q50 = np.quantile(arr.reshape(-1, 3), 0.50, axis=0).astype(np.float32)
            q75 = np.quantile(arr.reshape(-1, 3), 0.75, axis=0).astype(np.float32)
            feature = np.concatenate([mean, std, q25, q50, q75]).astype(np.float32)
    except Exception:
        feature = np.zeros(15, dtype=np.float32)
    cache[sample_id] = feature
    return feature


def load_or_build_image_feature_cache(
    manifest: pd.DataFrame,
    path: Path,
    *,
    image_size: int,
    refresh: bool,
) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    requested_ids = manifest["sample_id"].astype(str).tolist()
    if path.exists() and not refresh:
        loaded = np.load(path, allow_pickle=False)
        ids = loaded["sample_ids"].astype(str).tolist()
        features = loaded["features"].astype(np.float32)
        cache = {sample_id: features[idx] for idx, sample_id in enumerate(ids)}
        missing = [sample_id for sample_id in requested_ids if sample_id not in cache]
        if not missing:
            print(f"[build_graph] loaded image feature cache: {path} ({len(cache)} samples)", flush=True)
            return cache
        print(f"[build_graph] image cache missing {len(missing)} samples; rebuilding: {path}", flush=True)

    cache: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    ids: list[str] = []
    for idx, (_, row) in enumerate(manifest.iterrows(), start=1):
        feature = image_feature(row, cache, image_size=image_size)
        ids.append(str(row["sample_id"]))
        features.append(feature)
        if idx % 10000 == 0:
            print(f"[build_graph] cached image stats {idx}/{len(manifest)}", flush=True)
    feature_array = np.vstack(features).astype(np.float32) if features else np.empty((0, 15), dtype=np.float32)
    np.savez(path, sample_ids=np.asarray(ids, dtype=str), features=feature_array)
    print(f"[build_graph] wrote image feature cache: {path} ({len(ids)} samples)", flush=True)
    return {sample_id: feature_array[idx] for idx, sample_id in enumerate(ids)}


def make_patient_nodes(df: pd.DataFrame, lesion_x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    patients = sorted(df["patient_id"].fillna("UNKNOWN_PATIENT").astype(str).unique().tolist())
    features = []
    for patient in patients:
        mask = df["patient_id"].fillna("UNKNOWN_PATIENT").astype(str).to_numpy() == patient
        features.append(lesion_x[mask].mean(axis=0))
    return np.vstack(features).astype(np.float32) if features else np.empty((0, lesion_x.shape[1]), dtype=np.float32), patients


def make_attribute_nodes(
    graph_df: pd.DataFrame,
    train_df: pd.DataFrame,
    lesion_x: np.ndarray,
    feature_dim: int,
) -> tuple[np.ndarray, list[str]]:
    train_attrs = set()
    for _, row in train_df.iterrows():
        train_attrs.update(row_attributes(row))
    attrs = sorted(train_attrs | {"sex=UNK", "anatom_site=UNK", "age_bin=age=UNK", "size_bin=size=UNK"})
    features = []
    graph_attrs = [row_attributes(row) for _, row in graph_df.iterrows()]
    for attr in attrs:
        mask = np.asarray([attr in values for values in graph_attrs], dtype=bool)
        features.append(lesion_x[mask].mean(axis=0) if mask.any() else np.zeros(feature_dim, dtype=np.float32))
    return np.vstack(features).astype(np.float32), attrs


def row_attributes(row: pd.Series) -> list[str]:
    return [
        f"sex={clean_attr(row.get('sex'))}",
        f"anatom_site={clean_attr(row.get('anatom_site'))}",
        f"age_bin={clean_attr(row.get('age_bin'))}",
        f"size_bin={clean_attr(row.get('size_bin'))}",
    ]


def clean_attr(value: object) -> str:
    text = str(value) if value is not None else "UNK"
    return text if text and text.lower() not in {"nan", "none"} else "UNK"


def patient_edges(df: pd.DataFrame, patients: list[str], offset: int) -> list[tuple[int, int, int, float]]:
    patient_to_idx = {patient: offset + idx for idx, patient in enumerate(patients)}
    rel = RELATION_NAMES.index("same_patient")
    edges = []
    for idx, patient in enumerate(df["patient_id"].fillna("UNKNOWN_PATIENT").astype(str).tolist()):
        patient_idx = patient_to_idx[patient]
        edges.append((idx, patient_idx, rel, 1.0))
        edges.append((patient_idx, idx, rel, 1.0))
    return edges


def attribute_edges(df: pd.DataFrame, attrs: list[str], offset: int) -> list[tuple[int, int, int, float]]:
    attr_to_idx = {attr: offset + idx for idx, attr in enumerate(attrs)}
    rel = RELATION_NAMES.index("has_attribute")
    edges = []
    for idx, (_, row) in enumerate(df.iterrows()):
        for attr in row_attributes(row):
            target = attr if attr in attr_to_idx else attr.split("=")[0] + "=UNK"
            if target not in attr_to_idx:
                continue
            attr_idx = attr_to_idx[target]
            edges.append((idx, attr_idx, rel, 1.0))
            edges.append((attr_idx, idx, rel, 1.0))
    return edges


def knn_edges(
    features: np.ndarray,
    *,
    train_size: int,
    k: int,
    relation: int,
    split: str,
    device: str,
    chunk_size: int,
) -> list[tuple[int, int, int, float]]:
    if train_size < 2 or k <= 0:
        return []
    if device == "cuda":
        return torch_knn_edges(features, train_size=train_size, k=k, relation=relation, split=split, chunk_size=chunk_size)
    return sklearn_knn_edges(features, train_size=train_size, k=k, relation=relation, split=split)


def sklearn_knn_edges(features: np.ndarray, *, train_size: int, k: int, relation: int, split: str) -> list[tuple[int, int, int, float]]:
    from sklearn.neighbors import NearestNeighbors

    train_features = features[:train_size]
    n_neighbors = min(k + 1, train_size)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(train_features)
    query = train_features if split == "train" else features
    distances, indices = nn.kneighbors(query)
    edges = []
    query_start = 0
    for local_i, (row_dist, row_idx) in enumerate(zip(distances, indices)):
        src = query_start + local_i
        if split != "train" and src < train_size:
            continue
        for dist, dst in zip(row_dist, row_idx):
            if src == int(dst):
                continue
            weight = max(0.0, 1.0 - float(dist))
            edges.append((src, int(dst), relation, weight))
            edges.append((int(dst), src, relation, weight))
    return edges


def torch_knn_edges(
    features: np.ndarray,
    *,
    train_size: int,
    k: int,
    relation: int,
    split: str,
    chunk_size: int,
) -> list[tuple[int, int, int, float]]:
    if train_size < 2 or k <= 0:
        return []
    device = torch.device("cuda")
    train_np = normalize_rows(features[:train_size].astype(np.float32))
    query_start = 0 if split == "train" else train_size
    query_np = normalize_rows(features[query_start:].astype(np.float32))
    if len(query_np) == 0:
        return []

    train_tensor = torch.from_numpy(train_np).to(device)
    topk = min(k + 1 if split == "train" else k, train_size)
    edges: list[tuple[int, int, int, float]] = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, len(query_np), chunk_size):
            end = min(start + chunk_size, len(query_np))
            query_tensor = torch.from_numpy(query_np[start:end]).to(device)
            similarity = query_tensor @ train_tensor.T
            values, indices = torch.topk(similarity, k=topk, dim=1, largest=True)
            values_cpu = values.cpu().numpy()
            indices_cpu = indices.cpu().numpy()
            for local_i, (row_values, row_indices) in enumerate(zip(values_cpu, indices_cpu)):
                src = query_start + start + local_i
                added = 0
                for value, dst in zip(row_values, row_indices):
                    dst_i = int(dst)
                    if src == dst_i:
                        continue
                    weight = max(0.0, float(value))
                    edges.append((src, dst_i, relation, weight))
                    edges.append((dst_i, src, relation, weight))
                    added += 1
                    if added >= k:
                        break
            del query_tensor, similarity, values, indices
    return edges


def resolve_knn_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        print("[build_graph] using CUDA KNN", flush=True)
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("--knn-device cuda requested, but torch.cuda.is_available() is false.")
    print("[build_graph] using CPU sklearn KNN", flush=True)
    return "cpu"


def prototype_edges(features: np.ndarray, prototypes: np.ndarray, offset: int, relation: int, *, topk: int) -> list[tuple[int, int, int, float]]:
    if len(prototypes) == 0:
        return []
    feature_norm = normalize_rows(features)
    proto_norm = normalize_rows(prototypes)
    similarity = feature_norm @ proto_norm.T
    k = min(topk, prototypes.shape[0])
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    edges = []
    for src, proto_ids in enumerate(top):
        for proto_id in proto_ids:
            weight = max(0.0, float(similarity[src, proto_id]))
            dst = offset + int(proto_id)
            edges.append((src, dst, relation, weight))
            edges.append((dst, src, relation, weight))
    return edges


def make_kmeans_prototypes(features: np.ndarray, n: int, *, seed: int, feature_dim: int) -> np.ndarray:
    if len(features) == 0 or n <= 0:
        return np.empty((0, feature_dim), dtype=np.float32)
    n_clusters = min(n, len(features))
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=min(2048, max(32, len(features))))
        km.fit(features)
        return km.cluster_centers_.astype(np.float32)
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), size=n_clusters, replace=False)
        return features[idx].astype(np.float32)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    denom[denom < 1e-8] = 1.0
    return values / denom


def cap_split_ids(manifest: pd.DataFrame, split_ids: dict[str, list[str]], max_samples: int, *, seed: int) -> dict[str, list[str]]:
    capped: dict[str, list[str]] = {}
    for split, ids in split_ids.items():
        if len(ids) <= max_samples:
            capped[split] = ids
            continue
        part = manifest[manifest["sample_id"].astype(str).isin(ids)]
        pos = part[part["target"] == 1]
        neg = part[part["target"] == 0]
        n_pos = min(len(pos), max(1, max_samples // 2))
        n_neg = min(len(neg), max_samples - n_pos)
        sampled = pd.concat(
            [
                pos.sample(n=n_pos, random_state=seed) if n_pos else pos.iloc[[]],
                neg.sample(n=n_neg, random_state=seed + 1) if n_neg else neg.iloc[[]],
            ]
        )
        capped[split] = sampled["sample_id"].astype(str).tolist()
    return capped


def relation_counts(edge_type: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(edge_type == idx)) for idx, name in enumerate(RELATION_NAMES)}


def write_nodes_edges_tables(graph: dict[str, object], out_dir: Path, fold: int, split: str) -> None:
    nodes = pd.DataFrame({"node_id": range(len(graph["node_name"])), "node_type": graph["node_type"], "node_name": graph["node_name"]})
    nodes.to_parquet(out_dir / f"fold{fold}_{split}_nodes.parquet", index=False)
    edge_index = graph["edge_index"].numpy()
    edges = pd.DataFrame(
        {
            "src": edge_index[0],
            "dst": edge_index[1],
            "edge_type": [RELATION_NAMES[i] for i in graph["edge_type"].numpy().tolist()],
            "edge_weight": graph["edge_weight"].numpy(),
        }
    )
    edges.to_parquet(out_dir / f"fold{fold}_{split}_edges.parquet", index=False)


if __name__ == "__main__":
    main()
