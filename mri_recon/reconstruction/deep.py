from warnings import warn
import torch
import deepinv as dinv

class RAMReconstructor(dinv.models.Reconstructor):
    """
    Wrapper for RAM.
    Normalises input by magnitude of adjoint.

    :param float default_sigma: default sigma for RAM input. Overriden if physics already has a sigma (e.g. in a Gaussian noise model) at inference time.
    """
    def __init__(self, default_sigma=0.05):
        super().__init__()
        self.ram = dinv.models.RAM()
        self.default_sigma = default_sigma

    def forward(self, y, physics):
        _x_adj = physics.A_adjoint(y)
        scale = torch.quantile(_x_adj, 0.99)
        
        physics_norm = physics.compute_norm(torch.randn_like(_x_adj)).item()
        physics_adjointness = physics.adjointness_test(torch.randn_like(_x_adj)).item()

        if physics_norm > 1.2 or physics_norm < 0.8:
            raise ValueError(f"RAM reconstructor requires physics norm = 1 but got {physics_norm:.4f}")
        if physics_adjointness > 0.1 or physics_adjointness < -0.1:
            raise ValueError(f"RAM reconstructor requires physics adjointness = 0 but got {physics_adjointness:.4f}")

        sigma = None if hasattr(physics, "noise_model") and hasattr(physics.noise_model, "sigma") else self.default_sigma

        with torch.no_grad():
            return self.ram(y / scale, physics, sigma=sigma) * scale