from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.visualize_dataset_html import get_dataset_info, visualize_dataset_html


if __name__ == "__main__":
    init_logging()

    # 参数配置
    data_root = Path(__file__).resolve().parents[1] / "lerobot_datasets"
    username = "hhws"
    dataset_name = "so100_bimanual_transfer_4"
    load_from_hf_hub = False

    repo_id = f"{username}/{dataset_name}"
    dataset = None
    if repo_id:
        dataset = (
            LeRobotDataset(repo_id, root=data_root / repo_id) if not load_from_hf_hub else get_dataset_info(repo_id)
        )

    visualize_dataset_html(dataset)
