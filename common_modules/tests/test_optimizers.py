# ================= BERT PRETRAINING UTILS =================
import torch as t
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
import torch as t
import torch.nn.functional as F
import torch_optimizer as optim

class Net(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)


def _get_moon_data(unsqueeze_y=False):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = t.tensor(X, dtype=t.float32)
    y = t.tensor(y, dtype=t.int64)
    if unsqueeze_y:
        y = y.unsqueeze(-1)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)


def _train_with_opt(model, opt):
    dl = _get_moon_data()
    for i, (X, y) in enumerate(dl):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()


def test_adam(Adam):

    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)


def test_lamb(LAMBSimple):

    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = optim.Lamb(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = LAMBSimple(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        print(f"actual: {w0_correct}")
        print(f"submitted: {w0_submitted}")
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)
