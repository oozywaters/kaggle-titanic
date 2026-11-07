"""PyTorch MLP classifier with a scikit-learn estimator interface."""

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# single-threaded torch avoids a segfault from the duplicate OpenMP runtimes
# that torch and catboost/lightgbm/xgboost each bundle on macOS
torch.set_num_threads(1)

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """MLP for tabular classification, usable inside an sklearn Pipeline.

    Architecture and training are fully driven by constructor arguments:
    hidden layer sizes, activation, batchnorm, dropout, optimizer, learning
    rate, weight decay, cosine scheduler, epochs and batch size.
    """

    def __init__(
        self,
        hidden=(64, 32),
        activation="relu",
        batchnorm=True,
        dropout=0.0,
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        scheduler="none",
        epochs=60,
        batch_size=64,
        random_state=42,
    ):
        self.hidden = hidden
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _build_network(self, n_features, n_classes):
        layers = []
        in_size = n_features
        for out_size in self.hidden:
            layers.append(nn.Linear(in_size, out_size))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(out_size))
            layers.append(ACTIVATIONS[self.activation]())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_size = out_size
        layers.append(nn.Linear(in_size, n_classes))
        return nn.Sequential(*layers)

    @staticmethod
    def _to_dense(X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return X.astype(np.float32)

    def fit(self, X, y):
        X = self._to_dense(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)

        torch.manual_seed(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)

        self.network_ = self._build_network(X.shape[1], len(self.classes_))
        optimizer = OPTIMIZERS[self.optimizer](
            self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = None
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        self.network_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.network_(xb), yb)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return self

    def predict_proba(self, X):
        X = self._to_dense(X)
        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
