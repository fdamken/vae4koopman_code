import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import matplotlib.pyplot as plt
import numpy as np
import sacred
import torch

from investigation.plot_util import tuda, figsize
from src import util
from src.util import ExperimentNotConfiguredInterrupt

jsonpickle_numpy.register_handlers()
util.apply_sacred_frame_error_workaround()
torch.set_default_dtype(torch.double)


class Encoder(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self._pipe = torch.nn.Sequential(
            torch.nn.Linear(in_features, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, out_features)
        )

    def forward(self, x):
        return self._pipe(x)


class Decoder(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self._pipe = torch.nn.Sequential(
            torch.nn.Linear(in_features, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, out_features)
        )

    def forward(self, x):
        return self._pipe(x)


class LearnableMatrix(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self._pipe = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self._pipe(x)

    def weight(self):
        return self._pipe.weight


ex = sacred.Experiment('deep-koopman-lusch')
ex.observers.append(sacred.observers.FileStorageObserver('tmp_results_benchmarking'))


@ex.config
def defaults():
    latent_dim = None
    data_file = None

    alpha_1 = None
    alpha_2 = None
    alpha_3 = None
    S_p = None


# noinspection PyUnusedLocal
@ex.named_config
def pendulum_gym():
    latent_dim = 4
    data_file = 'tmp_data/pendulum_gym.json'

    alpha_1 = 0.001
    alpha_2 = 1e-9
    alpha_3 = 1e-14
    S_p = 30


@ex.automain
def main(latent_dim, data_file, alpha_1, alpha_2, alpha_3, S_p):
    if data_file is None:
        raise ExperimentNotConfiguredInterrupt()

    with open(data_file) as file:
        config = jsonpickle.loads(file.read())
        data = config['data']
    T = config['T']
    T_train = config['T_train']
    h = config['h']
    observation_dim_names = config['observation_dim_names']
    observations = data['observations']
    observations_noisy = data['observations_noisy']
    observations_train = data['observations_train']
    N, _, observation_dim = observations_train.shape

    encoder = Encoder(observation_dim, latent_dim).to('cuda')
    decoder = Decoder(latent_dim, observation_dim).to('cuda')
    K_single = LearnableMatrix(latent_dim, latent_dim).to('cuda')
    K = lambda m: torch.nn.Sequential(*([K_single] * m))

    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
        {'params': K_single.parameters()}
    ], lr=0.001)

    iteration = 1
    x = torch.tensor(observations_train[0, :, :], device='cuda')
    while True:
        x_phi = encoder(x)

        optimizer.zero_grad()

        L_recon = ((x[0] - decoder(x_phi[0])) ** 2).sum().sqrt()
        L_pred = 0.0
        for m in range(S_p):
            L_pred += ((x[m + 1] - decoder(K(m + 1)(x_phi[m]))) ** 2).sum().sqrt()
        L_pred /= S_p
        L_lin = 0.0
        for m in range(T_train - 1):
            L_lin += ((x_phi[m + 1] - K(m + 1)(x_phi[m])) ** 2).sum().sqrt()
        L_lin /= T_train - 1
        L_infty = (x[0] - decoder(x_phi[0])).abs().max() + (x[1] - decoder(K(1)(x_phi[0]))).abs().max()
        L = alpha_1 * (L_recon + L_pred) + L_lin + alpha_2 * L_infty

        L.backward()
        optimizer.step()

        print('Iteration %5d; Loss: %f' % (iteration, L))

        if iteration >= 50:
            break

        iteration += 1

    K_mat = K_single.weight()
    latent_rollout = [encoder(x[0])]
    rollout = [decoder(latent_rollout[-1])]
    for t in range(1, T):
        latent_rollout.append(K_mat @ latent_rollout[-1])
        rollout.append(decoder(latent_rollout[-1]))
    rollout = np.asarray([p.detach().cpu().numpy() for p in rollout])
    rollout_train = rollout[:T_train]
    rollout_test = rollout[T_train - 1:]

    domain = np.arange(T) * h
    domain_train = domain[:T_train]
    domain_test = domain[T_train - 1:]
    plot_noisy_data = not np.allclose(observations_noisy, observations)
    fig, axs = plt.subplots(observation_dim, 1, figsize=figsize(observation_dim, 1), squeeze=True)
    for dim, (dim_name, ax) in enumerate(zip(observation_dim_names, axs)):
        # Ground truth.
        ax.scatter(domain, observations[0, :, dim], s=1, color=tuda('black'), label='Truth')
        if plot_noisy_data:
            ax.scatter(domain, observations_noisy[0, :, dim], s=1, color=tuda('black'), alpha=0.2, label='Truth (Noisy)')

        # Rollout.
        ax.plot(domain_train, rollout_train[:, dim], color=tuda('blue'), label='Rollout')
        ax.plot(domain_test, rollout_test[:, dim], color=tuda('blue'), ls='dashed', label='Rollout (Prediction)')

        # Prediction boundary and learned initial value.
        ax.axvline(domain_train[-1], color=tuda('red'), ls='dotted', label='Prediction Boundary')

        if dim == 0:
            pass
        if dim == observation_dim - 1:
            ax.set_xlabel('Time Steps')
        ax.set_ylabel(dim_name)
        ax.legend()
    fig.show()
