import torch
import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms
from neurogrow.model import DynamicWidthCNNLightning


class TinyFakeDataModule(pl.LightningDataModule):
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


@pytest.fixture
def model():
    return DynamicWidthCNNLightning(
        initial_width=8, learning_rate=0.001, width_double_epoch=2
    )


@pytest.fixture
def datamodule():
    return TinyFakeDataModule(batch_size=2, num_workers=0)


def test_training_step(model, datamodule):
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_validation_step(model, datamodule):
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))
    model.validation_step(batch, 0)  # Should run without error


def test_test_step(model, datamodule):
    datamodule.setup()
    batch = next(iter(datamodule.test_dataloader()))
    model.test_step(batch, 0)  # Should run without error


def test_configure_optimizers(model):
    opt = model.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_width_doubling(model):
    old_width = model.width
    model.double_width()
    assert model.width == old_width * 2
