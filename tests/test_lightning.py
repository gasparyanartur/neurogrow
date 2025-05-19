import os
import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from neurogrow.model import DynamicWidthCNNLightning
from neurogrow.data import CIFAR10DataModule


class FakeImageNetDataModule(pl.LightningDataModule):
    """A fake data module for testing that uses FakeData instead of real ImageNet."""

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
                size=8,
                image_size=(3, 224, 224),
                num_classes=1000,
                transform=self.transform,
            )
            self.val_dataset = FakeData(
                size=4,
                image_size=(3, 224, 224),
                num_classes=1000,
                transform=self.transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = FakeData(
                size=4,
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
    """Fixture that provides a model instance for testing."""
    return DynamicWidthCNNLightning(
        initial_width=16,  # Small width for faster testing
        learning_rate=0.001,
        width_double_epoch=2,
    )


@pytest.fixture
def datamodule():
    """Fixture that provides a fake data module for testing."""
    return FakeImageNetDataModule(batch_size=2, num_workers=0)


def test_model_initialization(model):
    """Test model initialization with different widths."""
    assert model.width == 16
    assert model.conv1.out_channels == 16
    assert model.conv2.out_channels == 32
    assert model.conv3.out_channels == 64
    assert model.fc2.out_features == 1000  # ImageNet classes


def test_model_forward_pass(model):
    """Test that the forward pass works and returns correct shape."""
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)  # ImageNet image size
    output = model(input_tensor)

    # Check output shape (batch_size, num_classes)
    assert output.shape == (batch_size, 1000)
    # Check that output contains finite values
    assert torch.isfinite(output).all()


def test_model_width_doubling(model):
    """Test that width doubling works correctly."""
    old_width = model.width
    model.double_width()

    assert model.width == old_width * 2
    assert model.conv1.out_channels == old_width * 2
    assert model.conv2.out_channels == old_width * 4
    assert model.conv3.out_channels == old_width * 8


def test_training_step(model, datamodule):
    """Test that a single training step works."""
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))

    # Run training step
    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_validation_step(model, datamodule):
    """Test that validation step works."""
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))
    # Just check that it runs without error
    model.validation_step(batch, 0)


def test_test_step(model, datamodule):
    """Test that test step works."""
    datamodule.setup()
    batch = next(iter(datamodule.test_dataloader()))
    # Just check that it runs without error
    model.test_step(batch, 0)


def test_training_loop(model, datamodule):
    """Test that a full training loop works and width doubles as expected."""
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=datamodule)
    # After 1 epoch, width should not be doubled yet (still 16)
    assert model.width == 16


def test_model_checkpointing(model, datamodule, tmp_path):
    """Test that model checkpointing works."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path,
        filename="test-model-{epoch:02d}",
        save_top_k=1,
        monitor="val_loss",
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)
    # Check that checkpoint was saved
    assert len(list(tmp_path.glob("*.ckpt"))) > 0
    # Test loading checkpoint
    loaded_model = DynamicWidthCNNLightning.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    assert loaded_model.width == model.width
    assert loaded_model.learning_rate == model.learning_rate
