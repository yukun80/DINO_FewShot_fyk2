
import os
import json
import random
import argparse
from glob import glob

"""
* 当 --query 参数被设置为一个具体的正数时，脚本的行为和现在一样，选择一个固定大小的查询集。
* 当 --query 参数被设置为 -1 时，脚本将自动把所有未被选为支持集的样本全部用作查询集。

--run:
python -m datasets.generate_disaster_splits --path ../_datasets/Exp_Disaster_Few-Shot --shots 20 --query 257
"""

def generate_splits(dataset_path, shots, num_query):
    """
    Generates a JSON file with support and query splits for the disaster dataset.

    The script assumes the following directory structure:
    <dataset_path>/
    └── valset/
        ├── images/
        │   ├── *.tif
        └── labels/
            ├── *.tif

    Args:
        dataset_path (str): Absolute path to the Exp_Disaster_Few-Shot directory.
        shots (int): Number of samples for the support set.
        num_query (int): Number of samples for the query set.
    """
    print(f"Generating splits for {shots}-shot learning...")
    print(f"Dataset path: {dataset_path}")

    val_image_dir = os.path.join(dataset_path, 'valset', 'images')
    val_label_dir = os.path.join(dataset_path, 'valset', 'labels')

    if not os.path.isdir(val_image_dir):
        raise FileNotFoundError(f"Image directory not found at: {val_image_dir}")
    if not os.path.isdir(val_label_dir):
        raise FileNotFoundError(f"Label directory not found at: {val_label_dir}")

    # Find all image files and create corresponding label file paths
    all_images = sorted(glob(os.path.join(val_image_dir, '*.tif')))
    all_labels = [os.path.join(val_label_dir, os.path.basename(p)) for p in all_images]

    if len(all_images) < shots + num_query:
        raise ValueError(
            f"Not enough images in the dataset for the requested split. "
            f"Found {len(all_images)}, but need {shots} for support and {num_query} for query."
        )

    # Shuffle the dataset to ensure random selection
    random.seed(42) # Use a fixed seed for reproducibility
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    # Create support and query sets
    support_images = list(all_images[:shots])
    support_labels = list(all_labels[:shots])
    query_images = list(all_images[shots:shots + num_query])
    query_labels = list(all_labels[shots:shots + num_query])

    # The paths in the JSON should be relative to the project root for portability
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    def make_relative(path_list):
        return [os.path.relpath(p, project_root) for p in path_list]

    split_data = {
        'support': {
            'images': make_relative(support_images),
            'labels': make_relative(support_labels)
        },
        'query': {
            'images': make_relative(query_images),
            'labels': make_relative(query_labels)
        }
    }

    # Save the JSON file in the datasets directory
    output_filename = os.path.join(os.path.dirname(__file__), f'disaster_{shots}shot_splits.json')
    with open(output_filename, 'w') as f:
        json.dump(split_data, f, indent=4)

    print(f"Successfully generated splits.")
    print(f"Support set size: {len(support_images)}")
    print(f"Query set size: {len(query_images)}")
    print(f"Split file saved to: {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Few-Shot Data Splits")
    parser.add_argument('--path', type=str, required=True,
                        help='Absolute path to the Exp_Disaster_Few-Shot dataset directory.')
    parser.add_argument('--shots', type=int, default=10,
                        help='Number of support samples.')
    parser.add_argument('--query', type=int, default=40,
                        help='Number of query samples.')
    args = parser.parse_args()

    generate_splits(args.path, args.shots, args.query)
