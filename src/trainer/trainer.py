"""
Training and evaluation pipeline.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.trainer.losses import get_loss
from src.trainer.metrics import get_metrics, DiceMetric
from src.models.build import save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer class for segmentation models.

    Handles training, validation, and inference.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        logger: Optional[Any] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            logger: Logger instance
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        # Training settings
        self.epochs = config["training"]["epochs"]
        self.device = self._get_device()

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup loss
        self.criterion = get_loss(config)

        # Setup metrics
        self.metrics = get_metrics(config)

        # Mixed precision
        self.use_amp = config["hardware"].get("mixed_precision", False)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.accumulation_steps = config["training"].get("accumulation_steps", 1)

        # Output directory
        self.output_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}

        # Resume from checkpoint
        if resume_from:
            self._resume(resume_from)

    def _get_device(self) -> torch.device:
        """Get compute device."""
        device_str = self.config["hardware"]["device"]

        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        opt_config = self.config["training"]["optimizer"]
        opt_name = opt_config["name"].lower()

        params = self.model.parameters()
        lr = opt_config["lr"]
        weight_decay = opt_config.get("weight_decay", 0)

        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            betas = tuple(opt_config.get("betas", [0.9, 0.999]))
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        elif opt_name == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        sched_config = self.config["training"].get("scheduler", {})
        sched_name = sched_config.get("name", "cosine").lower()

        if sched_name == "cosine":
            warmup = sched_config.get("warmup_epochs", 0)
            min_lr = sched_config.get("min_lr", 1e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - warmup, eta_min=min_lr
            )
        elif sched_name == "step":
            step_size = sched_config.get("step_size", 30)
            gamma = sched_config.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif sched_name == "plateau":
            patience = sched_config.get("patience", 10)
            factor = sched_config.get("factor", 0.1)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=patience, factor=factor
            )
        else:
            return None

    def _resume(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        checkpoint = load_checkpoint(self.model, checkpoint_path)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]

        if "best_metric" in checkpoint:
            self.best_metric = checkpoint["best_metric"]

        if self.logger:
            self.logger.info(f"Resumed from epoch {self.current_epoch}")

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Training history
        """
        early_stop_config = self.config["training"].get("early_stopping", {})
        patience = early_stop_config.get("patience", 30)
        no_improve_count = 0

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_metrics = self._validate()
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_metrics.get("dice", 0))

            # Log
            if self.logger:
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Val Loss: {val_loss:.4f} "
                    f"Val Dice: {val_metrics.get('dice', 0):.4f}"
                )

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("dice", 0))
                else:
                    self.scheduler.step()

            # Save checkpoints
            self._save_checkpoints(val_metrics)

            # Early stopping
            if val_metrics.get("dice", 0) > self.best_metric:
                self.best_metric = val_metrics.get("dice", 0)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if early_stop_config.get("enabled", False) and no_improve_count >= patience:
                if self.logger:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return self.history

    def _train_epoch(self) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})

        return total_loss / num_batches

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        # Reset metrics
        dice_metric = DiceMetric(num_classes=self.config["model"]["out_channels"])

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                # Update metrics
                preds = torch.argmax(outputs, dim=1)
                dice_metric.update(preds, labels)

        # Compute metrics
        metrics = dice_metric.compute()

        return total_loss / num_batches, metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation/test set."""
        _, metrics = self._validate()
        return metrics

    def predict(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """
        Run inference on new data.

        Args:
            input_path: Input data directory
            output_path: Output directory for predictions
        """
        import nibabel as nib
        import numpy as np

        self.model.eval()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get input files
        modalities = self.config["data"]["modalities"]

        # Find all cases
        cases = set()
        for mod in modalities:
            mod_dir = input_path / mod.lower()
            if mod_dir.exists():
                for f in mod_dir.iterdir():
                    if f.suffix in [".nii", ".gz"]:
                        cases.add(f.stem.replace(".nii", ""))

        # Process each case
        with torch.no_grad():
            for case_id in tqdm(cases, desc="Inference"):
                # Load modalities
                images = []
                affine = None

                for mod in modalities:
                    mod_path = input_path / mod.lower() / f"{case_id}.nii.gz"
                    if not mod_path.exists():
                        mod_path = input_path / mod.lower() / f"{case_id}.nii"

                    if mod_path.exists():
                        nii = nib.load(str(mod_path))
                        images.append(nii.get_fdata().astype(np.float32))
                        if affine is None:
                            affine = nii.affine

                if len(images) != len(modalities):
                    continue

                # Stack and prepare input
                image = np.stack(images, axis=0)  # [C, H, W, D]
                image = torch.from_numpy(image).unsqueeze(0).to(self.device)  # [1, C, H, W, D]

                # Run inference (sliding window for large volumes)
                output = self._sliding_window_inference(image)

                # Get prediction
                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                # Save
                pred_nii = nib.Nifti1Image(pred.astype(np.uint8), affine)
                nib.save(pred_nii, str(output_path / f"{case_id}_pred.nii.gz"))

    def _sliding_window_inference(self, image: torch.Tensor) -> torch.Tensor:
        """
        Sliding window inference for large volumes.

        Args:
            image: Input tensor [B, C, H, W, D]

        Returns:
            Predictions [B, num_classes, H, W, D]
        """
        try:
            from monai.inferers import sliding_window_inference

            roi_size = self.config["inference"]["sliding_window"]["roi_size"]
            overlap = self.config["inference"]["sliding_window"]["overlap"]

            return sliding_window_inference(
                image,
                roi_size=tuple(roi_size),
                sw_batch_size=self.config["inference"]["batch_size"],
                predictor=self.model,
                overlap=overlap,
            )
        except ImportError:
            # Fallback to direct inference
            return self.model(image)

    def _save_checkpoints(self, metrics: Dict[str, float]) -> None:
        """Save model checkpoints."""
        ckpt_config = self.config["training"].get("checkpoint", {})

        # Save last
        if ckpt_config.get("save_last", True):
            save_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                str(self.output_dir / "last.pth"),
                best_metric=self.best_metric,
                history=self.history,
            )

        # Save best
        if ckpt_config.get("save_best", True):
            if metrics.get("dice", 0) >= self.best_metric:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.current_epoch,
                    str(self.output_dir / "best.pth"),
                    best_metric=metrics.get("dice", 0),
                    history=self.history,
                )

        # Save every N epochs
        save_every = ckpt_config.get("save_every", 0)
        if save_every > 0 and (self.current_epoch + 1) % save_every == 0:
            save_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                str(self.output_dir / f"epoch_{self.current_epoch+1}.pth"),
                best_metric=self.best_metric,
            )
