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