from warnings import warn
import torch
import deepinv as dinv

class RAMReconstructor(dinv.models.Reconstructor):
    """
    Wrapper for RAM from DeepInverse.
    Normalises input by magnitude of adjoint.

    :param float default_sigma: default sigma for RAM input. Overriden if physics already has a sigma (e.g. in a Gaussian noise model) at inference time.
    """
    def __init__(self, default_sigma=0.05, device: torch.device = None) -> None:
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
        self.device = device
        
        self.model = dinv.models.RAM(device=device)
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
            return self.model(y / scale, physics, sigma=sigma) * scale

class DeepImagePriorReconstructor(dinv.models.Reconstructor):
    """
    Wrapper for Deep Image Prior from DeepInverse.
    
    :param tuple img_size: image size of the output. Defaults to (640, 368)
    :param int n_iter: number of iterations to fit the DIP. Defaults to 100.
    """
    def __init__(self, img_size: tuple = (640, 368), n_iter: int = 100,) -> None:
        super().__init__()
        
        lr = 1e-2  # learning rate for the optimizer.
        channels = 64  # number of channels per layer in the decoder.
        in_size = [2, 2]  # size of the input to the decoder.

        self.model = dinv.models.DeepImagePrior(
            dinv.models.ConvDecoder(
                img_size=(2, *img_size), in_size=in_size, channels=channels
            ),
            learning_rate=lr,
            iterations=n_iter,
            verbose=True,
            input_size=[channels] + in_size,
        )

    def forward(self, y, physics):
        return self.model(y, physics)