import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from neurogrow.model import DynamicWidthCNNLightning
from neurogrow.data import CIFAR10DataModule


def main():
    parser = argparse.ArgumentParser(
        description="Train ImageNet CNN with dynamic width"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="path to ImageNet dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="input batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--initial-width", type=int, default=32, help="initial network width"
    )
    parser.add_argument(
        "--width-double-epoch",
        type=int,
        default=5,
        help="epoch interval for doubling width",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="directory to save model checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="directory to save tensorboard logs",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="accelerator to use (auto, cpu, gpu, etc.)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="number of devices to use",
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=None,
        help="number of batches to overfit on",
    )
    args = parser.parse_args()

    # Create save and log directories
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data module
    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Initialize model
    model = DynamicWidthCNNLightning(
        initial_width=args.initial_width,
        learning_rate=args.lr,
        width_double_epoch=args.width_double_epoch,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="imagenet-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
    )

    # Setup logger
    logger = TensorBoardLogger(log_dir, name="imagenet")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        overfit_batches=args.overfit_batches,
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Test the model
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
