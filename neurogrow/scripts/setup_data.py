from torchvision import datasets

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Setup data")
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="path to data directory"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download CIFAR10
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)


if __name__ == "__main__":
    main()
