import torch
from torch import nn

print('hi')

a1 = torch.tensor(4.0, requires_grad=True)
a2 = torch.tensor(5.0)

a3 = torch.tensor(2.0, requires_grad=True)

c1 = (a1 * a2) / a3
c2 = (a1 * a2) / a3
c3 = (a1 * a2) / a3

c1.backward()
print(a1.grad)

c2.backward()
print(a1.grad)

c3.backward()
print(a1.grad)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(1, 2),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(2, 2),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, x, alpha=1.0):
        out1 = self.block1(x)
        out2 = self.classification_head(out1)

        return out2

criterion1 = nn.CrossEntropyLoss()

model = TestModel()
x1 = torch.Tensor([[[10, 11], [12, 13]]])
x2 = torch.Tensor([[[10, 11], [12, 13]]])

print(x1.grad)
model(x1)
print(x1.grad)
model(x2)
print(x1.grad)