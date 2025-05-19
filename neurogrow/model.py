import pytorch_lightning as pl
from typing import Tuple
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from pytorch_lightning.utilities.types import STEP_OUTPUT


class DynamicWidthCNN(nn.Module):
    def __init__(self, initial_width=32):
        super(DynamicWidthCNN, self).__init__()
        self.width = initial_width
        self.conv1 = nn.Conv2d(1, self.width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.width, self.width * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.width * 2, self.width * 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.width * 4 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def double_width(self):
        old_width = self.width
        self.width *= 2

        # Create new layers with doubled width
        new_conv1 = nn.Conv2d(1, self.width, kernel_size=3, padding=1)
        new_conv2 = nn.Conv2d(self.width, self.width * 2, kernel_size=3, padding=1)
        new_conv3 = nn.Conv2d(self.width * 2, self.width * 4, kernel_size=3, padding=1)
        new_fc1 = nn.Linear(self.width * 4 * 3 * 3, 128)

        # Copy weights from old layers to new layers
        with torch.no_grad():
            # For conv1: [out_channels, in_channels, kernel_size, kernel_size]
            new_conv1.weight.data[:old_width] = self.conv1.weight.data
            new_conv1.bias.data[:old_width] = self.conv1.bias.data

            # For conv2: [out_channels, in_channels, kernel_size, kernel_size]
            new_conv2.weight.data[: old_width * 2, :old_width] = self.conv2.weight.data
            new_conv2.bias.data[: old_width * 2] = self.conv2.bias.data

            # For conv3: [out_channels, in_channels, kernel_size, kernel_size]
            new_conv3.weight.data[: old_width * 4, : old_width * 2] = (
                self.conv3.weight.data
            )
            new_conv3.bias.data[: old_width * 4] = self.conv3.bias.data

            # For fc1: [out_features, in_features]
            new_fc1.weight.data[:, : old_width * 4 * 3 * 3] = self.fc1.weight.data
            new_fc1.bias.data = self.fc1.bias.data

        # Replace old layers with new ones
        self.conv1 = new_conv1
        self.conv2 = new_conv2
        self.conv3 = new_conv3
        self.fc1 = new_fc1

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, self.width * 4 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DynamicWidthCNNLightning(pl.LightningModule):
    def __init__(
        self,
        initial_width: int = 32,
        learning_rate: float = 0.001,
        width_double_epoch: int | None = 5,
    ):
        super().__init__()
        # Save hyperparameters as class attributes for type checking
        self.initial_width = initial_width
        self.learning_rate = learning_rate
        self.width_double_epoch = width_double_epoch
        self.save_hyperparameters()

        # Initialize model components
        self.width = initial_width
        self.conv1 = nn.Conv2d(3, self.width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.width, self.width * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.width * 2, self.width * 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.width * 4 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1000)
        self.relu = nn.ReLU()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=1000)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=1000)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=1000)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, self.width * 4 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def double_width(self) -> None:
        old_width = self.width
        self.width *= 2

        # Create new layers with doubled width
        new_conv1 = nn.Conv2d(3, self.width, kernel_size=3, padding=1).to(
            device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
        )
        new_conv2 = nn.Conv2d(self.width, self.width * 2, kernel_size=3, padding=1).to(
            device=self.conv2.weight.device, dtype=self.conv2.weight.dtype
        )
        new_conv3 = nn.Conv2d(
            self.width * 2, self.width * 4, kernel_size=3, padding=1
        ).to(device=self.conv3.weight.device, dtype=self.conv3.weight.dtype)
        new_fc1 = nn.Linear(self.width * 4 * 28 * 28, 128).to(
            device=self.fc1.weight.device, dtype=self.fc1.weight.dtype
        )

        # Copy weights from old layers to new layers
        with torch.no_grad():
            new_conv1.weight.data[:old_width] = self.conv1.weight.data
            new_conv1.bias.data[:old_width] = self.conv1.bias.data

            new_conv2.weight.data[: old_width * 2, :old_width] = self.conv2.weight.data
            new_conv2.bias.data[: old_width * 2] = self.conv2.bias.data

            new_conv3.weight.data[: old_width * 4, : old_width * 2] = (
                self.conv3.weight.data
            )
            new_conv3.bias.data[: old_width * 4] = self.conv3.bias.data

            new_fc1.weight.data[:, : old_width * 4 * 28 * 28] = self.fc1.weight.data
            new_fc1.bias.data = self.fc1.bias.data

        # Replace old layers with new ones
        self.conv1 = new_conv1
        self.conv2 = new_conv2
        self.conv3 = new_conv3
        self.fc1 = new_fc1

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.val_accuracy(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("params", self.count_parameters(), on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.test_accuracy(logits, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, prog_bar=True)
        self.log("params", self.count_parameters(), on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        # Double width at specified intervals
        if self.width_double_epoch is None:
            return

        if self.current_epoch > 0 and self.current_epoch % self.width_double_epoch == 0:
            self.double_width()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
