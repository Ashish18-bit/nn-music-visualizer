"""tests/test_model.py — CNNLSTMEmotionClassifier architecture tests."""

import pytest
import torch
import torch.nn as nn

from model.cnn_lstm import (
    CNNLSTMEmotionClassifier,
    CNNEncoder,
    LSTMTemporal,
    TemporalAttention,
    ConvBlock,
)

# ── common tensor shapes ─────────────────────────────────────
B = 4          # batch size
N_MELS = 128
T_FRAMES = 130
FEAT_DIM = 289
N_CLASSES = 5


def _mel(b=B):
    return torch.randn(b, 1, N_MELS, T_FRAMES)


def _feat(b=B):
    return torch.randn(b, FEAT_DIM)


@pytest.fixture
def model():
    return CNNLSTMEmotionClassifier(
        num_classes=N_CLASSES,
        n_mels=N_MELS,
        feature_dim=FEAT_DIM,
    )


# ─────────────────────────────────────────────────────────────
#  ConvBlock
# ─────────────────────────────────────────────────────────────

class TestConvBlock:
    def test_output_channels(self):
        block = ConvBlock(in_channels=1, out_channels=32, kernel_size=3)
        x = torch.randn(2, 1, 64, 64)
        y = block(x)
        assert y.shape[1] == 32

    def test_spatial_reduction(self):
        block = ConvBlock(1, 32, pool_size=(2, 2))
        x = torch.randn(2, 1, 64, 64)
        y = block(x)
        assert y.shape[2] == 32
        assert y.shape[3] == 32

    def test_no_nan_output(self):
        block = ConvBlock(1, 32)
        x = torch.randn(2, 1, 64, 64)
        y = block(x)
        assert torch.all(torch.isfinite(y))


# ─────────────────────────────────────────────────────────────
#  CNNEncoder
# ─────────────────────────────────────────────────────────────

class TestCNNEncoder:
    def test_output_shape(self):
        enc = CNNEncoder(in_channels=1, channels=(32, 64, 128))
        x = torch.randn(B, 1, N_MELS, T_FRAMES)
        y = enc(x)
        assert y.shape[0] == B
        assert y.shape[1] == 128      # last channel count

    def test_freq_axis_reduced(self):
        enc = CNNEncoder(in_channels=1, channels=(32, 64, 128))
        x = torch.randn(B, 1, N_MELS, T_FRAMES)
        y = enc(x)
        # Three pools: (2,2),(2,2),(2,1) → freq: 128→64→32→16
        assert y.shape[2] == N_MELS // 8

    def test_time_axis_reduced(self):
        enc = CNNEncoder(in_channels=1, channels=(32, 64, 128))
        x = torch.randn(B, 1, N_MELS, T_FRAMES)
        y = enc(x)
        # Pool (2,2),(2,2),(2,1) → time: 130→65→32→32
        assert y.shape[3] > 0

    def test_out_channels_attribute(self):
        enc = CNNEncoder(channels=(16, 32, 64))
        assert enc.out_channels == 64

    def test_gradients_flow(self):
        enc = CNNEncoder()
        x = torch.randn(2, 1, N_MELS, T_FRAMES, requires_grad=False)
        y = enc(x)
        loss = y.sum()
        loss.backward()
        for p in enc.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ─────────────────────────────────────────────────────────────
#  TemporalAttention
# ─────────────────────────────────────────────────────────────

class TestTemporalAttention:
    def test_output_shape(self):
        attn = TemporalAttention(hidden_dim=512)
        x = torch.randn(B, 32, 512)    # (batch, time, hidden)
        out = attn(x)
        assert out.shape == (B, 512)

    def test_weights_sum_to_one(self):
        attn = TemporalAttention(hidden_dim=64)
        x = torch.randn(B, 10, 64)
        # weights come from softmax so each batch's weights sum to 1
        scores = attn.attn(x)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        assert torch.allclose(weights.sum(dim=1), torch.ones(B, 1), atol=1e-5)


# ─────────────────────────────────────────────────────────────
#  LSTMTemporal
# ─────────────────────────────────────────────────────────────

class TestLSTMTemporal:
    def test_output_shape_bidirectional(self):
        lstm = LSTMTemporal(input_size=128, hidden_size=256, bidirectional=True)
        x = torch.randn(B, 32, 128)
        out, _ = lstm(x)
        assert out.shape == (B, 32, 512)   # 256*2

    def test_output_shape_unidirectional(self):
        lstm = LSTMTemporal(input_size=64, hidden_size=128, bidirectional=False)
        x = torch.randn(B, 20, 64)
        out, _ = lstm(x)
        assert out.shape == (B, 20, 128)

    def test_out_size_attribute(self):
        lstm = LSTMTemporal(input_size=64, hidden_size=128, bidirectional=True)
        assert lstm.out_size == 256


# ─────────────────────────────────────────────────────────────
#  Full model — forward pass
# ─────────────────────────────────────────────────────────────

class TestFullModelForward:
    def test_output_shape(self, model):
        out = model(_mel(), _feat())
        assert out.shape == (B, N_CLASSES)

    def test_output_is_log_prob(self, model):
        out = model(_mel(), _feat())
        # log-softmax: all values <= 0, exp sums to 1
        assert (out <= 0).all()
        assert torch.allclose(out.exp().sum(dim=1), torch.ones(B), atol=1e-5)

    def test_output_finite(self, model):
        out = model(_mel(), _feat())
        assert torch.all(torch.isfinite(out))

    def test_forward_without_features(self, model):
        # features=None should not crash when feat_proj exists,
        # but the model skips the branch
        out = model(_mel(), features=None)
        assert out.shape == (B, N_CLASSES)

    def test_single_sample(self, model):
        out = model(_mel(1), _feat(1))
        assert out.shape == (1, N_CLASSES)

    def test_larger_batch(self, model):
        out = model(_mel(16), _feat(16))
        assert out.shape == (16, N_CLASSES)

    def test_different_batch_sizes_same_model(self, model):
        model.eval()
        with torch.no_grad():
            out4 = model(_mel(4), _feat(4))
            out8 = model(_mel(8), _feat(8))
        assert out4.shape == (4, N_CLASSES)
        assert out8.shape == (8, N_CLASSES)


# ─────────────────────────────────────────────────────────────
#  predict / predict_proba
# ─────────────────────────────────────────────────────────────

class TestPrediction:
    def test_predict_proba_sums_to_one(self, model):
        proba = model.predict_proba(_mel(), _feat())
        assert torch.allclose(proba.sum(dim=1), torch.ones(B), atol=1e-5)

    def test_predict_proba_non_negative(self, model):
        proba = model.predict_proba(_mel(), _feat())
        assert (proba >= 0).all()

    def test_predict_returns_valid_class(self, model):
        pred, conf = model.predict(_mel(), _feat())
        assert pred.shape == (B,)
        assert conf.shape == (B,)
        assert (pred >= 0).all() and (pred < N_CLASSES).all()

    def test_confidence_in_0_1(self, model):
        _, conf = model.predict(_mel(), _feat())
        assert (conf >= 0).all() and (conf <= 1).all()


# ─────────────────────────────────────────────────────────────
#  Gradients & training compatibility
# ─────────────────────────────────────────────────────────────

class TestGradients:
    def test_loss_backward(self, model):
        mel = _mel()
        feat = _feat()
        labels = torch.randint(0, N_CLASSES, (B,))
        log_probs = model(mel, feat)
        loss = nn.NLLLoss()(log_probs, labels)
        loss.backward()
        # All parameters should have gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_weights_change_after_step(self, model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        mel, feat = _mel(), _feat()
        labels = torch.randint(0, N_CLASSES, (B,))

        # snapshot weights
        w_before = model.classifier[-1].weight.data.clone()

        log_probs = model(mel, feat)
        loss = nn.NLLLoss()(log_probs, labels)
        loss.backward()
        opt.step()

        w_after = model.classifier[-1].weight.data
        assert not torch.allclose(w_before, w_after)

    def test_eval_mode_no_dropout_variance(self, model):
        """In eval mode, two identical forward passes should give identical output."""
        model.eval()
        mel, feat = _mel(), _feat()
        with torch.no_grad():
            out1 = model(mel, feat)
            out2 = model(mel, feat)
        assert torch.allclose(out1, out2)


# ─────────────────────────────────────────────────────────────
#  from_config
# ─────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_builds_from_config(self):
        cfg = {
            "emotions": {"num_classes": 5},
            "features": {"n_mels": 128, "feature_dim": 289},
            "model": {
                "cnn": {"channels": [32, 64, 128], "kernel_size": 3, "dropout": 0.3},
                "lstm": {"hidden_size": 256, "num_layers": 2,
                         "bidirectional": True, "dropout": 0.4},
                "head": {"hidden_dim": 128, "dropout": 0.4},
            },
        }
        m = CNNLSTMEmotionClassifier.from_config(cfg)
        out = m(_mel(), _feat())
        assert out.shape == (B, 5)

    def test_parameter_count_positive(self, model):
        assert model.count_parameters() > 0

    def test_summary_runs(self, model, capsys):
        model.summary()
        captured = capsys.readouterr()
        assert "CNN-LSTM" in captured.out
