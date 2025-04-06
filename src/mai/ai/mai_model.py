from torch import nn


class MAINet(nn.Module):
    def __init__(self, modules_amount: int):
        assert isinstance(modules_amount, int)
        assert modules_amount > 1
        super().__init__()
        from mai.capnp.data_classes import STATE_KEYS
        self.fc1 = nn.Linear(len(STATE_KEYS), 256)
        self.fc2 = nn.Linear(256, modules_amount)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x
