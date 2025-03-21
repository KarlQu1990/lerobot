import datasets
from pathlib import Path

root = Path("/home/hhws/projects/robot_datasets/hhws/so100_bimanual_transfer_6/data/chunk-000/")
for f in root.iterdir():
    ds = datasets.Dataset.from_parquet(str(f),split="train")
    features = ds.features
    ds = ds.to_dict()
    # ds["task_index"] = [0] * len(ds["task_index"])
    ds["episode_index"] = [int(f.stem.split("_")[1])] * len(ds["task_index"])

    ds = datasets.Dataset.from_dict(ds, features=features, split="train")
    ds.to_parquet(str(f))
