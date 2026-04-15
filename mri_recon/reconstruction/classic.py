import deepinv as dinv

class ZeroFilledReconstructor(dinv.models.Reconstructor):
    def forward(self, y, physics):
        return physics.A_adjoint(y)

class ConjugateGradientReconstructor(dinv.models.Reconstructor):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, y, physics):
        return physics.A_dagger(y, **self.kwargs)

class TVPGDReconstructor(dinv.models.Reconstructor):
    """
    Proximal Gradient Descent algorithm regularised by Total Variation (TV) regulariser.

    :param int n_iter: max number of iterations to run PGD. Defaults to 100.
    """
    def __init__(self, n_iter: int = 100, **kwargs):
        super().__init__()

        self.model = dinv.optim.PGD(
            prior=dinv.optim.prior.TVPrior(n_it_max=100),
            data_fidelity=dinv.optim.data_fidelity.L2(),
            stepsize=1.0,
            lambda_reg=1e-2,  # TV regularisation parameter,
            early_stop=True,
            max_iter=n_iter,
            verbose=True,
            thres_conv=1e-7,
        )

    def forward(self, y, physics):
        out, metrics = self.model(y, physics, compute_metrics=True)
        dinv.utils.plotting.plot_curves(metrics)
        return out
    
class WaveletFISTAReconstructor(dinv.models.Reconstructor):
    """
    FISTA algorithm regularised by wavelet regulariser.

    :param int n_iter: max number of iterations to run FISTA. Defaults to 100.
    """
    def __init__(self, n_iter: int = 100, device="cpu", **kwargs):
        super().__init__()

        self.model = dinv.optim.FISTA(
            prior=dinv.optim.prior.WaveletPrior(level=4, wv="db8", p=1, device=device),
            data_fidelity=dinv.optim.data_fidelity.L2(),
            stepsize=0.2,
            lambda_reg=1e-3,
            early_stop=True,
            max_iter=n_iter,
            verbose=True,
            thres_conv=1e-7,
        ).to(device)

    def forward(self, y, physics):
        out, metrics = self.model(y, physics, compute_metrics=True)
        dinv.utils.plotting.plot_curves(metrics)
        return out

class TVFISTAReconstructor(dinv.models.Reconstructor):
    """
    FISTA algorithm regularised by Total Variation (TV) regulariser.

    :param int n_iter: max number of iterations to run FISTA. Defaults to 100.
    """
    def __init__(self, n_iter: int = 100, **kwargs):
        super().__init__()

        self.model = dinv.optim.FISTA(
            prior=dinv.optim.prior.TVPrior(n_it_max=100),
            data_fidelity=dinv.optim.data_fidelity.L2(),
            stepsize=1.0,
            lambda_reg=1e-2,  # TV regularisation parameter,
            early_stop=True,
            max_iter=n_iter,
            verbose=True,
            thres_conv=1e-7,
        )

    def forward(self, y, physics):
        out, metrics = self.model(y, physics, compute_metrics=True)
        dinv.utils.plotting.plot_curves(metrics)
        return out