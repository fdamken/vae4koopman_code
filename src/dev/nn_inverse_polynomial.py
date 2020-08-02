import shutil
import tempfile
from argparse import ArgumentParser

import torch
from neptunecontrib.monitoring.sacred import NeptuneObserver
from sacred import Experiment
from sacred.run import Run


ex = Experiment('nn_inverse_polynomial')
ex.observers.append(NeptuneObserver(project_name = 'fdamken/variational-koopman'))



class ForwardModel(torch.nn.Module):
    in_features = 2
    out_features = 3


    def __init__(self):
        super().__init__()

        self._A = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype = torch.float32)
        self._B = torch.tensor([[0, 0], [0, 0], [1, 0]], dtype = torch.float32)


    def forward(self, x):
        # This is equivalent to \( [x_1, x_2, x_1^2] \).
        return torch.einsum('ij,bj->bi', self._A, x) + torch.einsum('ij,bj->bi', self._B, x * x)



class InverseModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self._pipe = torch.nn.Sequential(
                torch.nn.Linear(in_features, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, out_features),
                torch.nn.ReLU()
        )


    def forward(self, x):
        return self._pipe(x)



@ex.config
def config():
    learning_rate = 0.01
    loss_threshold = 1e-5
    save_model_every_n_iterations = 10



@ex.capture
def save_model(phi_inverse: InverseModel, loss_train_item: float, iteration: int, /, out_dir: str, _run: Run):
    model_file_name = '%s/nn_inverse_polynomial_model_%036.30f_%05d.pkl' % (out_dir, loss_train_item, iteration)
    torch.save(phi_inverse.state_dict(), model_file_name)
    _run.add_artifact(model_file_name, metadata = { 'iteration': iteration, 'loss_train': loss_train_item })



@ex.automain
def main(_run: Run, learning_rate: float, loss_threshold: float, save_model_every_n_iterations: int):
    parser = ArgumentParser()
    parser.add_argument('--cuda', action = 'store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    torch.random.manual_seed(42)

    domain = torch.arange(0, 0.5, 0.001)
    domain_grid = torch.meshgrid([domain, domain])
    samples = torch.cat([x.reshape(-1, 1) for x in domain_grid], dim = 1).to(device = device)

    phi = ForwardModel().to(device = device)
    phi_inverse = InverseModel(phi.out_features, phi.in_features).to(device = device)

    train_in = phi(samples)
    train_out = samples.clone()

    optimizer = torch.optim.Adam(params = phi_inverse.parameters(), lr = learning_rate)
    loss_fn = torch.nn.MSELoss()

    out_dir = tempfile.mkdtemp()

    iteration = 1
    loss_train = None
    while loss_train is None or loss_train > loss_threshold:
        optimizer.zero_grad()
        loss_train = loss_fn(phi_inverse(train_in), train_out)
        loss_train.backward()
        optimizer.step()

        print('Iter. %5d: Loss (Train): %.30f' % (iteration, loss_train.item()))
        if iteration % save_model_every_n_iterations == 0:
            save_model(phi_inverse, loss_train.item(), iteration, out_dir = out_dir)

        _run.log_scalar('loss_train', loss_train, iteration)

        iteration += 1

    save_model(phi_inverse, loss_train.item(), iteration - 1, out_dir = out_dir)

    shutil.rmtree(out_dir)
