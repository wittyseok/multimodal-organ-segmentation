"""
t-SNE visualization for feature analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class TSNEVisualizer:
    """
    t-SNE visualization for analyzing feature representations.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_layer: str = "encoder",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
    ):
        """
        Initialize t-SNE visualizer.

        Args:
            model: PyTorch model
            feature_layer: Layer to extract features from
            n_components: Number of t-SNE dimensions
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
        """
        self.model = model
        self.feature_layer = feature_layer
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter

        self.features = None
        self._register_hook()

    def _register_hook(self) -> None:
        """Register hook to extract features."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.features = output[0].detach()
            else:
                self.features = output.detach()

        for name, module in self.model.named_modules():
            if self.feature_layer in name:
                module.register_forward_hook(hook)
                break

    def extract_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataset.

        Args:
            dataloader: Data loader
            device: Device to use
            max_samples: Maximum number of samples

        Returns:
            Tuple of (features, labels)
        """
        self.model.eval()
        self.model.to(device)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                images = batch["image"].to(device)
                labels = batch.get("label", None)

                # Forward pass to trigger hook
                _ = self.model(images)

                # Get extracted features
                if self.features is not None:
                    # Global average pooling if 3D features
                    if self.features.dim() > 2:
                        pooled = self.features.mean(dim=tuple(range(2, self.features.dim())))
                    else:
                        pooled = self.features

                    all_features.append(pooled.cpu().numpy())

                    if labels is not None:
                        # Use mode of labels for each sample
                        for i in range(labels.shape[0]):
                            label_mode = labels[i].flatten().mode()[0].item()
                            all_labels.append(label_mode)

                if max_samples and len(all_features) * dataloader.batch_size >= max_samples:
                    break

        features = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels) if all_labels else None

        return features, labels

    def compute_tsne(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute t-SNE embedding.

        Args:
            features: Feature vectors [N, D]

        Returns:
            t-SNE embedding [N, n_components]
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("scikit-learn is required for t-SNE")

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            random_state=42,
        )

        embedding = tsne.fit_transform(features)

        return embedding

    def visualize(
        self,
        embedding: np.ndarray,
        labels: Optional[np.ndarray] = None,
        class_names: Optional[Dict[int, str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "t-SNE Visualization",
    ) -> None:
        """
        Visualize t-SNE embedding.

        Args:
            embedding: t-SNE embedding [N, 2]
            labels: Class labels [N]
            class_names: Mapping from label to name
            save_path: Path to save figure
            title: Figure title
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                name = class_names.get(label, str(label)) if class_names else str(label)
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[color],
                    label=name,
                    alpha=0.7,
                    s=20,
                )

            ax.legend(loc="best", fontsize=8)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()

    def run(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        max_samples: int = 1000,
        class_names: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Full t-SNE pipeline: extract features, compute t-SNE, visualize.

        Args:
            dataloader: Data loader
            save_path: Path to save visualization
            device: Device to use
            max_samples: Maximum samples to use
            class_names: Class name mapping

        Returns:
            t-SNE embedding
        """
        # Extract features
        features, labels = self.extract_features(dataloader, device, max_samples)

        # Compute t-SNE
        embedding = self.compute_tsne(features)

        # Visualize
        self.visualize(embedding, labels, class_names, save_path)

        return embedding
