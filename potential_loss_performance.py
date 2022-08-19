#%%
import time
from functools import wraps

import torch
from dataclasses import dataclass

from torch.nn.functional import mse_loss

T = torch.tensor


@dataclass
class TestData:
    beta: T
    x: T
    particle_id: T
    pred: T
    truth: T


def generate_test_data(
        n_nodes=1000, n_particles=250, n_x_features=3, rng=None
    ) -> TestData:
        if rng is None:
            rng = np.random.default_rng()
        return TestData(
            beta=torch.from_numpy(rng.random(n_nodes)),
            x=torch.from_numpy(rng.random((n_nodes, n_x_features))),
            particle_id=torch.from_numpy(rng.choice(np.arange(n_particles), size=n_nodes)),
            pred=torch.from_numpy(rng.choice([0., 1.], size=(n_nodes, 1))),
            truth=torch.from_numpy(rng.choice([0., 1.], size=(n_nodes, 1))),
        )



import numpy as np

# original:  0.40115785598754883  s
# vectorized:  0.13051199913024902 s
class ObjectLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, sb=0.1, device="cpu", mode="efficiency"):
        super().__init__()
        #: Strength of noise suppression
        self.sb = sb
        self.q_min = q_min
        self.device = device
        self.mode = mode
        #: Scale up loss value by this factor
        self.scale = 100

    def MSE(self, p, t):
        return torch.sum(mse_loss(p, t, reduction="none"), dim=1)

    # @profile
    def object_loss(self, *, pred, beta, truth, particle_id):
        noise_mask = particle_id == 0
        # shape: n_nodes
        xi = (~noise_mask) * torch.arctanh(beta) ** 2
        # shape: n_nodes
        mse = self.MSE(pred, truth)
        if self.mode == "purity":
            return self.scale / torch.sum(xi) * torch.mean(xi * mse)
        # shape: n_particles
        pids = torch.unique(particle_id[particle_id > 0])
        # PID masks (n_nodes x n_particles)
        masks = particle_id[:, None] == pids[None, :]
        # shape: (n_nodes x n_particles)
        xi_ps = masks * (torch.arctanh(beta) ** 2)[:, None]
        # shape: n_nodes
        weights = 1.0 / (torch.sum(xi_ps, dim=0))
        # shape: n_nodes
        facs = torch.sum(mse[:, None] * xi_ps, dim=0)
        loss = torch.mean(weights * facs)
        return self.scale * loss

    def forward(self, W, beta, H, pred, Y, particle_id, track_params, reconstructable):
        mask = reconstructable > 0
        return self.object_loss(
            pred=pred[mask],
            beta=beta[mask],
            truth=track_params[mask],
            particle_id=particle_id[mask],
        )

def timeit(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("@timefn: {} took {} seconds.".format(func.__name__, end_time - start_time))
        return result
    return measure_time

@timeit
def benchmark():
    pl = ObjectLoss()
    for i in range(100):
        td = generate_test_data()
        pl.object_loss(beta=td.beta, particle_id=td.particle_id, pred=td.pred, truth=td.truth)


benchmark()