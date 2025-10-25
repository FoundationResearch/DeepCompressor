import os
import random

import datasets
import yaml

__all__ = ["get_dataset"]


def _resolve_yaml_path(meta_path: str) -> str:
    if os.path.isabs(meta_path) and os.path.exists(meta_path):
        return meta_path
    # Try as given (relative to CWD)
    if os.path.exists(meta_path):
        return meta_path
    # Compute repo root from this file: deepcompressor/app/diffusion/dataset/data/__init__.py → repo root
    prefix = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(prefix, "../../../../../"))
    # Candidates to probe
    candidates = [
        os.path.join(os.getcwd(), meta_path),
        os.path.join(repo_root, meta_path),
        os.path.join(repo_root, "examples/diffusion", meta_path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return meta_path  # let open(...) raise a clear error


def load_dataset_yaml(meta_path: str, max_dataset_size: int = -1, repeat: int = 4) -> dict:
    meta_path = _resolve_yaml_path(meta_path)
    meta = yaml.safe_load(open(meta_path, "r"))
    names = list(meta.keys())
    if max_dataset_size > 0:
        random.Random(0).shuffle(names)
        names = names[:max_dataset_size]
        names = sorted(names)

    ret = {"filename": [], "prompt": [], "meta_path": []}
    idx = 0
    for name in names:
        prompt = meta[name]
        for j in range(repeat):
            ret["filename"].append(f"{name}-{j}")
            ret["prompt"].append(prompt)
            ret["meta_path"].append(meta_path)
            idx += 1
    return ret


def get_dataset(
    name: str,
    config_name: str | None = None,
    split: str = "train",
    max_dataset_size: int = -1,
    return_gt: bool = False,
    repeat: int = 4,
    chunk_start: int = 0,
    chunk_step: int = 1,
) -> datasets.Dataset:
    prefix = os.path.dirname(__file__)
    kwargs = {
        "name": config_name,
        "split": split,
        "trust_remote_code": True,
        "token": True,
        "max_dataset_size": max_dataset_size,
    }
    if name.endswith((".yaml", ".yml")):
        dataset = datasets.Dataset.from_dict(
            load_dataset_yaml(name, max_dataset_size=max_dataset_size, repeat=repeat),
            features=datasets.Features(
                {
                    "filename": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "meta_path": datasets.Value("string"),
                }
            ),
        )
    else:
        path = os.path.join(prefix, f"{name}")
        if name == "COCO":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        elif name == "DCI":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        elif name == "MJHQ":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
    assert not hasattr(dataset, "_unchunk_size")
    assert not hasattr(dataset, "_chunk_start")
    assert not hasattr(dataset, "_chunk_step")
    unchunk_size = len(dataset)
    if chunk_step > 1 or chunk_start > 0:
        assert 0 <= chunk_start < chunk_step
        dataset = dataset.select(range(chunk_start, len(dataset), chunk_step))
    else:
        chunk_start, chunk_step = 0, 1
    dataset._unchunk_size = unchunk_size
    dataset._chunk_start = chunk_start
    dataset._chunk_step = chunk_step
    return dataset
