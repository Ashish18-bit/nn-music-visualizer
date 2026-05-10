"""
model/cnn_lstm.py
─────────────────
Hybrid CNN-LSTM Emotion Classifier.

Architecture
────────────

    Mel-spectrogram                Feature vector
    (B, 1, n_mels, T)              (B, feature_dim)
          │                               │
          ▼                               │
    ┌─────────────────┐                   │
    │  CNN Encoder    │                   │
    │  3 Conv Blocks  │                   │
    │  (Conv→BN→ReLU  │                   │
    │   →MaxPool)     │                   │
    └────────┬────────┘                   │
             │ (B, 128, 1, T')            │
             ▼                            │
    ┌─────────────────┐                   │
    │  Reshape to     │                   │
    │  (B, T', 128)   │                   │
    └────────┬────────┘                   │
             │                            │
             ▼                            │
    ┌─────────────────────────────────────┤
    │  Bidirectional LSTM                 │
    │  2 layers, hidden=256               │
    │  input: CNN features (128)          │
    │  output: (B, T', 512)               │
    └─────────────────┬───────────────────┘
                      │
              Temporal attention
              pooling → (B, 512)
                      │
                      ▼
             ┌────────────────┐
             │  Feature proj  │◄── feature_vector (B, feature_dim)
             │  (concat +     │
             │   Dense 256)   │
             └───────┬────────┘
                     │
                     ▼
             ┌────────────────┐
             │  Classifier    │
             │  Dense 128     │
             │  → ReLU        │
             │  → Dropout     │
             │  → Dense 5     │
             │  → LogSoftmax  │
             └────────────────┘

Key design decisions
────────────────────
- CNN extracts local spectro-temporal patterns (timbre, texture)
- BiLSTM captures how emotion evolves across the 3-second window
- Attention pooling focuses on the most informative time steps
- Feature vector branch fuses hand-crafted acoustic features
  (MFCCs, chroma, etc.) via a learned projection layer
- LogSoftmax + NLLLoss = numerically equivalent to CrossEntropyLoss
  but makes probabilities easier to inspect
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Single CNN encoder block:
      Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: Tuple[int, int] = (2, 2),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalAttention(nn.Module):
    """
    Scaled dot-product self-attention over the time axis.

    Input  : (B, T, hidden)
    Output : (B, hidden)  — weighted sum over T

    Learns to focus on the most emotionally salient time steps.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = self.attn(x)           # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = (x * weights).sum(dim=1)  # (B, H)
        return context


class CNNEncoder(nn.Module):
    """
    Three-block CNN that reads a mel-spectrogram and produces
    a sequence of spatial feature maps.

    Input  : (B, 1, n_mels, T)
    Output : (B, C_out, freq', time') where C_out = channels[-1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        ch_in = in_channels
        # Pool only the frequency axis in the last block to preserve time steps
        pool_sizes = [(2, 2), (2, 2), (2, 1)]
        for i, ch_out in enumerate(channels):
            layers.append(
                ConvBlock(ch_in, ch_out, kernel_size, pool_sizes[i], dropout)
            )
            ch_in = ch_out
        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LSTMTemporal(nn.Module):
    """
    Bidirectional LSTM that processes the CNN feature sequence.

    Input  : (B, T, input_size)
    Output : (B, T, hidden_size * 2)  — bidirectional
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(x)


# ─────────────────────────────────────────────────────────────
#  Full model
# ─────────────────────────────────────────────────────────────

class CNNLSTMEmotionClassifier(nn.Module):
    """
    Full hybrid CNN-LSTM emotion classifier.

    Parameters
    ----------
    num_classes    : number of emotion categories (default 5)
    n_mels         : mel-spectrogram frequency bins
    cnn_channels   : output channel counts for each CNN block
    cnn_kernel     : CNN kernel size
    cnn_dropout    : CNN spatial dropout rate
    lstm_hidden    : LSTM hidden state size (per direction)
    lstm_layers    : number of LSTM layers
    lstm_dropout   : LSTM dropout rate
    lstm_bidir     : whether LSTM is bidirectional
    head_hidden    : classifier head hidden layer size
    head_dropout   : classifier head dropout rate
    feature_dim    : acoustic feature vector dimension (0 = disable)

    Example
    -------
    model = CNNLSTMEmotionClassifier()
    mel   = torch.randn(8, 1, 128, 130)   # batch of mel-specs
    feats = torch.randn(8, 289)            # batch of feature vectors
    log_probs = model(mel, feats)          # (8, 5)
    """

    def __init__(
        self,
        num_classes: int = 5,
        n_mels: int = 128,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        cnn_kernel: int = 3,
        cnn_dropout: float = 0.3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.4,
        lstm_bidir: bool = True,
        head_hidden: int = 128,
        head_dropout: float = 0.4,
        feature_dim: int = 289,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # ── CNN encoder ──────────────────────────────────────
        self.cnn = CNNEncoder(
            in_channels=1,
            channels=cnn_channels,
            kernel_size=cnn_kernel,
            dropout=cnn_dropout,
        )

        # After 3 MaxPool(2,2)/(2,2)/(2,1), the freq axis is
        # reduced: n_mels → n_mels//2 → n_mels//4 → n_mels//8
        # Time axis reduced:  T → T//2 → T//4 (last pool is (2,1))
        cnn_freq_out = n_mels // (2 * 2 * 2)   # 128 → 16
        lstm_input_size = cnn_channels[-1] * cnn_freq_out

        # ── BiLSTM ───────────────────────────────────────────
        self.lstm = LSTMTemporal(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=lstm_bidir,
        )

        # ── Temporal attention ───────────────────────────────
        self.attention = TemporalAttention(self.lstm.out_size)

        # ── Feature vector projection ────────────────────────
        lstm_out = self.lstm.out_size  # 512 if bidirectional
        if feature_dim > 0:
            self.feat_proj = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=head_dropout),
            )
            combined_dim = lstm_out + 128
        else:
            self.feat_proj = None
            combined_dim = lstm_out

        # ── Classifier head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        self._init_weights()

    # ── forward ─────────────────────────────────────────────

    def forward(
        self,
        mel: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mel      : (B, 1, n_mels, T)
        features : (B, feature_dim) optional

        Returns
        -------
        log_probs : (B, num_classes)  — use with NLLLoss
        """
        B = mel.size(0)

        # ── CNN: spatial feature extraction ─────────────────
        cnn_out = self.cnn(mel)
        # cnn_out: (B, C, freq', time')

        # Flatten freq dimension → (B, C*freq', time')
        # then transpose → (B, time', C*freq') for LSTM
        C, Fq, T = cnn_out.shape[1], cnn_out.shape[2], cnn_out.shape[3]
        x = cnn_out.view(B, C * Fq, T)   # (B, C*freq', time')
        x = x.permute(0, 2, 1)           # (B, time', C*freq')

        # ── LSTM: temporal modelling ─────────────────────────
        lstm_out, _ = self.lstm(x)       # (B, time', lstm_out)

        # ── Attention pooling ────────────────────────────────
        context = self.attention(lstm_out)   # (B, lstm_out)

        # ── Feature vector branch ────────────────────────────
        if self.feat_proj is not None and features is not None:
            feat_out = self.feat_proj(features)      # (B, 128)
            combined = torch.cat([context, feat_out], dim=1)  # (B, lstm_out+128)
        elif self.feat_proj is not None:
            # features not provided — substitute zeros so classifier dim matches
            feat_out = torch.zeros(B, 128, device=context.device, dtype=context.dtype)
            combined = torch.cat([context, feat_out], dim=1)
        else:
            combined = context

        # ── Classification ────────────────────────────────────
        logits = self.classifier(combined)           # (B, num_classes)
        return F.log_softmax(logits, dim=1)

    def predict_proba(
        self,
        mel: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return class probabilities (softmax, not log-softmax)."""
        with torch.no_grad():
            log_probs = self.forward(mel, features)
        return log_probs.exp()

    def predict(
        self,
        mel: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted_class_idx, confidence) tensors."""
        proba = self.predict_proba(mel, features)
        confidence, pred = proba.max(dim=1)
        return pred, confidence

    # ── weight initialisation ────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    # ── utility ──────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        total = self.count_parameters()
        logger.info("CNNLSTMEmotionClassifier — %d trainable parameters", total)
        print(f"\n{'─'*55}")
        print(f"  CNN-LSTM Emotion Classifier")
        print(f"{'─'*55}")
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name:<20} {str(type(module).__name__):<25} {params:>8,} params")
        print(f"{'─'*55}")
        print(f"  {'Total':<46} {total:>8,} params")
        print(f"{'─'*55}\n")

    @classmethod
    def from_config(cls, cfg: dict) -> "CNNLSTMEmotionClassifier":
        """Build model from config.yaml model section."""
        m = cfg["model"]
        return cls(
            num_classes=cfg["emotions"]["num_classes"],
            n_mels=cfg["features"]["n_mels"],
            cnn_channels=tuple(m["cnn"]["channels"]),
            cnn_kernel=m["cnn"]["kernel_size"],
            cnn_dropout=m["cnn"]["dropout"],
            lstm_hidden=m["lstm"]["hidden_size"],
            lstm_layers=m["lstm"]["num_layers"],
            lstm_dropout=m["lstm"]["dropout"],
            lstm_bidir=m["lstm"]["bidirectional"],
            head_hidden=m["head"]["hidden_dim"],
            head_dropout=m["head"]["dropout"],
            feature_dim=cfg["features"]["feature_dim"],
        )
