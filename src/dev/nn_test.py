import numpy as np
import torch



class SimpleNeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        f_in = 2
        f_out = 3
        hidden = f_in * f_out * 10

        self._pipe = torch.nn.Sequential(
                torch.nn.Linear(f_in, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, f_out)
        )


    def forward(self, x):
        return self._pipe(x)



if __name__ == '__main__':
    torch.random.manual_seed(42)

    samples_in = torch.rand((1000, 2))
    samples_out = torch.tensor(np.asarray([(x[0], x[1], x[0] ** 2) for x in samples_in]))

    net = SimpleNeuralNet()
    optimizer = torch.optim.Adam(params = net.parameters(), lr = 0.01)
    loss_fn = torch.nn.MSELoss()

    iteration = 1
    loss = None
    while loss is None or loss > 1e-5:
        loss_previous = loss

        optimizer.zero_grad()
        loss = loss_fn(net(samples_in), samples_out)
        loss.backward()
        optimizer.step()

        print('Iter. %5d: Loss: %f' % (iteration, loss.item()))

        iteration += 1

    print(list(net.parameters()))
