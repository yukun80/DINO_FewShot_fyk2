import os
import json
import random
import argparse
from glob import glob
from typing import List, Tuple, Dict

"""
重构后的划分逻辑：
1) 仅从指定的“支持集地区列表”中抽取 shots 个样本作为支持集；
2) 查询集为除支持集之外的所有样本；
3) JSON 保存格式保持不变（support/query -> images/labels，且路径相对工程根目录）。

用法示例：
python -m datasets.generate_disaster_splits --path ../_datasets/Exp_Disaster_Few-Shot --shots 1
python -m datasets.generate_disaster_splits --path ../_datasets/Exp_Disaster_Few-Shot --shots 5

说明：
- 本脚本会忽略历史上的 --query 固定数量行为，统一使用“其余全部作为查询集”。
- 为保证可复现，随机种子固定为 42。
"""

# 指定支持集的地区列表（按需求固定）。
# 注意：如列表中存在重复项，会自动去重（避免无意义的重复偏置）。
SUPPORT_AREAS: List[str] = [
    "Asakura_Japan",
    "Askja_Iceland",
    "Chimanimani_Zimbabwe",
    "Kodagu_India",
    "Jiuzhaigou_China",
    "Los Lagos_Chile",
    "Osh_Kyrgyzstan",
    "Taitung_China",
    "Tbilisi_Georgia",
]


def _parse_area_from_filename(path: str) -> str:
    """根据文件名解析地区名（鲁棒到两个或多个坐标/编号后缀）。

    约定：valset 文件名满足 "<area>_<x>_<y>.tif" 或 "<area>_<n>.tif"，其中 <area> 可能包含空格或下划线。
    做法：去除文件名末尾所有“纯数字”的下划线分隔片段，剩余即为 area。

    示例：
    - Tbilisi_Georgia_12.tif              -> Tbilisi_Georgia
    - Los Lagos_Chile_003.tif             -> Los Lagos_Chile
    - Asakura_Japan_0_2048.tif            -> Asakura_Japan
    - Big Sur_United States_1024_4096.tif -> Big Sur_United States
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    # 从末尾剔除所有纯数字片段
    while parts and parts[-1].isdigit():
        parts.pop()
    # 兼容极端情况：若全部被弹出，则回退为原始 name
    area = "_".join(parts) if parts else name
    return area


def _collect_all_pairs(val_image_dir: str, val_label_dir: str) -> List[Tuple[str, str]]:
    """收集 (image_path, label_path) 对并按文件名排序。"""
    all_images = sorted(glob(os.path.join(val_image_dir, "*.tif")))
    all_labels = [os.path.join(val_label_dir, os.path.basename(p)) for p in all_images]
    return list(zip(all_images, all_labels))


def _select_support_from_areas(
    pairs: List[Tuple[str, str]],
    areas: List[str],
    shots: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """为每个指定地区各选择 shots 个样本作为支持集，其余为查询候选。

    规则：
    - 对 areas 去重（保持顺序）。
    - 跳过数据集中不存在的地区（打印告警）。
    - 对存在的每个地区，若候选数 < shots，报错（避免采样退化）。

    返回：(support_pairs, remaining_pairs)
    """
    # 去重并保序
    unique_areas: List[str] = list(dict.fromkeys(areas))

    # 将样本按地区聚合
    area_to_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for img, lbl in pairs:
        area = _parse_area_from_filename(img)
        area_to_pairs.setdefault(area, []).append((img, lbl))

    # 过滤出数据集中实际存在的地区，记录缺失项
    present_areas = [a for a in unique_areas if a in area_to_pairs]
    missing_areas = [a for a in unique_areas if a not in area_to_pairs]
    if missing_areas:
        print(f"Warning: 以下地区在数据集中未找到，将被跳过: {missing_areas}")

    if not present_areas:
        raise ValueError("在指定的 SUPPORT_AREAS 中，没有任何地区在数据集中出现。")

    # 校验每个存在的地区是否具备足够样本
    insufficient = {a: len(area_to_pairs[a]) for a in present_areas if len(area_to_pairs[a]) < shots}
    if insufficient:
        details = ", ".join([f"{a}({c})" for a, c in insufficient.items()])
        raise ValueError(f"以下地区可用样本不足（每个地区需要 {shots} 张）：{details}")

    # 固定随机种子，逐地区采样
    random.seed(42)
    support_pairs: List[Tuple[str, str]] = []
    for area in present_areas:
        candidates = area_to_pairs[area]
        # 为保证不同运行可复现，先对候选排序再采样
        candidates = sorted(candidates)
        support_pairs.extend(random.sample(candidates, shots))

    # 剩余样本 = 所有样本 - 支持样本
    support_set = set(support_pairs)
    remaining_pairs = [p for p in pairs if p not in support_set]

    return support_pairs, remaining_pairs


def _to_relative(split: List[str], project_root: str) -> List[str]:
    return [os.path.relpath(p, project_root) for p in split]


def generate_splits(dataset_path: str, shots: int) -> Dict[str, Dict[str, List[str]]]:
    """按照指定规则生成支持/查询划分，并保存 JSON。

    目录结构约定：
    <dataset_path>/
    └── valset/
        ├── images/*.tif
        └── labels/*.tif
    """
    print(f"Generating splits for {shots}-shot learning...")
    print(f"Dataset path: {dataset_path}")

    val_image_dir = os.path.join(dataset_path, "valset", "images")
    val_label_dir = os.path.join(dataset_path, "valset", "labels")

    if not os.path.isdir(val_image_dir):
        raise FileNotFoundError(f"Image directory not found at: {val_image_dir}")
    if not os.path.isdir(val_label_dir):
        raise FileNotFoundError(f"Label directory not found at: {val_label_dir}")

    # 收集所有样本对
    all_pairs = _collect_all_pairs(val_image_dir, val_label_dir)
    if len(all_pairs) == 0:
        raise FileNotFoundError("未在 valset/images 下找到任何 .tif 文件。")

    # 基于地区选择支持集；查询集为其余全部
    support_pairs, query_pairs = _select_support_from_areas(all_pairs, SUPPORT_AREAS, shots)

    # 输出信息：统计支持集的地区分布（便于快速核对）
    area_counts: Dict[str, int] = {}
    for img, _ in support_pairs:
        area = _parse_area_from_filename(img)
        area_counts[area] = area_counts.get(area, 0) + 1
    print(f"Support areas picked (per-area counts): {dict(sorted(area_counts.items()))}")

    # 转为 images/labels 列表
    support_images = [p[0] for p in support_pairs]
    support_labels = [p[1] for p in support_pairs]
    query_images = [p[0] for p in query_pairs]
    query_labels = [p[1] for p in query_pairs]

    # JSON 中保存相对工程根目录的路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_data = {
        "support": {
            "images": _to_relative(support_images, project_root),
            "labels": _to_relative(support_labels, project_root),
        },
        "query": {
            "images": _to_relative(query_images, project_root),
            "labels": _to_relative(query_labels, project_root),
        },
    }

    # 保存到 datasets 目录下，文件名保持不变格式
    output_filename = os.path.join(os.path.dirname(__file__), f"disaster_{shots}shot_splits.json")
    with open(output_filename, "w") as f:
        json.dump(split_data, f, indent=4)

    print("Successfully generated splits.")
    print(f"Support set size: {len(support_images)}")
    print(f"Query set size: {len(query_images)}")
    print(f"Split file saved to: {output_filename}")

    return split_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Few-Shot Data Splits (area-restricted support)")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Absolute path to the Exp_Disaster_Few-Shot dataset directory.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10,
        help="Number of support samples to draw from predefined areas.",
    )
    # 兼容旧参数，但已不再使用固定查询数量逻辑
    parser.add_argument(
        "--query",
        type=int,
        default=-1,
        help="Deprecated. Ignored. Query set is all remaining samples.",
    )
    args = parser.parse_args()

    generate_splits(args.path, args.shots)
