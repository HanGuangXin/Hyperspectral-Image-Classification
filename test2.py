import torch
from torch import nn
import torchsnooper

# example 1
# @torchsnooper.snoop()
# def myfunc(mask, x):
#     y = torch.zeros(6, device='cuda')
#     y.masked_scatter_(mask, x)
#     return y
#
# mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda')
# source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
# mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda', dtype=torch.uint8)
# y = myfunc(mask, source)

# example 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with torchsnooper.snoop():
    for _ in range(100):
        optimizer.zero_grad()
        # pred = model(x)
        pred = model(x).squeeze()
        squared_diff = (y - pred) ** 2
        loss = squared_diff.mean()
        print(loss.item())
        loss.backward()
        optimizer.step()