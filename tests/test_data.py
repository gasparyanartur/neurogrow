import torch
import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms
from neurogrow.data import CIFAR10DataModule


class TinyFakeImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = FakeData(
                size=4,
                image_size=(3, 224, 224),
                num_classes=1000,
                transform=self.transform,
            )
            self.val_dataset = FakeData(
                size=2,
                image_size=(3, 224, 224),
                num_classes=1000,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = FakeData(
                size=2,
                image_size=(3, 224, 224),
                num_classes=1000,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def test_imagenet_datamodule():
    dm = TinyFakeImageNetDataModule(batch_size=2)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Check train loader
    batch = next(iter(train_loader))
    assert batch[0].shape == (2, 3, 224, 224)
    assert batch[1].shape == (2,)
    assert batch[0].dtype == torch.float32
    assert batch[1].dtype == torch.long

    # Check val loader
    batch = next(iter(val_loader))
    assert batch[0].shape == (2, 3, 224, 224)
    assert batch[1].shape == (2,)

    # Check test loader
    batch = next(iter(test_loader))
    assert batch[0].shape == (2, 3, 224, 224)
    assert batch[1].shape == (2,)


def test_imagenet_data_range():
    dm = TinyFakeImageNetDataModule(batch_size=2)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    data = batch[0]
    # After normalization, mean should be close to 0 and std close to 1 (roughly, since FakeData is random)
    assert abs(data.mean().item()) < 0.5
    assert abs(data.std().item() - 1.0) < 0.5
