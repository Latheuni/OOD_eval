from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.util import copy_docs_from
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import Transform, constraints
from pyro.distributions.transforms.affine_autoregressive import AffineAutoregressive, affine_autoregressive
from torch import nn
import torch
import torch.nn.functional as F
import torch.distributions as tdist

# Code copied form: https://github.com/sharpenb/Posterior-Network/tree/main

## Normalizing flow classes
class NormalizingFlowDensity(nn.Module):

    def __init__(self, dim, flow_length, flow_type='planar_flow'):
        super(NormalizingFlowDensity, self).__init__()
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        if self.flow_type == 'radial_flow':
            self.transforms = nn.Sequential(*(
                Radial(dim) for _ in range(flow_length)
            ))
        elif self.flow_type == 'iaf_flow':
            self.transforms = nn.Sequential(*(
                affine_autoregressive(dim, hidden_dims=[128, 128]) for _ in range(flow_length)
            ))
        else:
            raise NotImplementedError

    def forward(self, z):

        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x

class MixtureDensity(nn.Module):

    def __init__(self, dim, n_components=20, mixture_type='normal_mixture'):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        self.mixture_type = mixture_type

        if self.mixture_type == 'normal_mixture':
            # Isotropic Gaussians
            self.log_pi = torch.nn.Parameter(torch.randn(self.n_components, 1))
            self.log_pi.requires_grad = True
            self.mu = torch.nn.Parameter(torch.randn(self.n_components, self.dim))
            self.mu.requires_grad = True
            self.log_sigma_ = torch.nn.Parameter(torch.randn(self.n_components, self.dim, self.dim)/100.)
            self.log_sigma_.requires_grad = True

            self.softmax = nn.Softmax(dim=-1)
        else:
            raise NotImplementedError

    def forward(self, x):
        pi = self.softmax(self.log_pi)
        # Parametrization with LL^T where diagonal of L are positives.
        sigma = self.log_sigma_ * torch.tril(torch.ones_like(self.log_sigma_))
        sigma = sigma - torch.diag_embed(torch.diagonal(sigma, dim1=-2, dim2=-1)) + torch.diag_embed(torch.diagonal(torch.exp(sigma), dim1=-2, dim2=-1)) + .001 * torch.diag_embed(torch.ones(self.n_components, self.dim)).to(x.device.type)
        dist = tdist.MultivariateNormal(loc=self.mu, scale_tril=sigma)

        expand_x = x.unsqueeze(1).repeat(1, self.n_components, 1)
        p_expand_x = torch.exp(dist.log_prob(expand_x))
        log_prob_x = torch.log(torch.matmul(p_expand_x, pi).squeeze())
        return log_prob_x

    def log_prob(self, x):
        return self.forward(x)

@copy_docs_from(Transform)
class ConditionedRadial(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, params):
        super().__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None

    # This method ensures that torch(u_hat, w) > -1, required for invertibility
    def u_hat(self, u, w):
        raise NotImplementedError()
        alpha = torch.matmul(u.unsqueeze(-2), w.unsqueeze(-1)).squeeze(-1)
        a_prime = -1 + F.softplus(alpha)
        return u + (a_prime - alpha) * w.div(w.pow(2).sum(dim=-1, keepdim=True))

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """
        x0, alpha_prime, beta_prime = self._params() if callable(self._params) else self._params

        # Ensure invertibility using approach in appendix A.2
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and logDet using Equation 14.
        diff = x - x0[:, None, :]
        r = diff.norm(dim=-1, keepdim=True).squeeze()
        h = (alpha[:, None] + r).reciprocal()
        h_prime = - (h ** 2)
        beta_h = beta[:, None] * h

        self._cached_logDetJ = ((x0.size(-1) - 1) * torch.log1p(beta_h) +
                                torch.log1p(beta_h + beta[:, None] * h_prime * r))
        return x + beta_h[:, :, None] * diff

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of
        inverting arbitrary values `y`; rather it assumes `y` is the result of a
        previously computed application of the bijector to some `x` (which was
        cached on the forward call)
        """

        raise KeyError("ConditionedRadial object expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_logDetJ


@copy_docs_from(ConditionedRadial)
class Radial(ConditionedRadial, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, c, input_dim):
        super().__init__(self._params)

        self.x0 = nn.Parameter(torch.Tensor(c, input_dim,))
        self.alpha_prime = nn.Parameter(torch.Tensor(c,))
        self.beta_prime = nn.Parameter(torch.Tensor(c,))
        self.c = c
        self.input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        return self.x0, self.alpha_prime, self.beta_prime

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.x0.size(1))
        self.alpha_prime.data.uniform_(-stdv, stdv)
        self.beta_prime.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)


class BatchedNormalizingFlowDensity(nn.Module):

    def __init__(self, c, dim, flow_length, flow_type='planar_flow'):
        super(BatchedNormalizingFlowDensity, self).__init__()
        self.c = c
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.mean = nn.Parameter(torch.zeros(self.c, self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim).repeat(self.c, 1, 1), requires_grad=False)

        if self.flow_type == 'radial_flow':
            self.transforms = nn.Sequential(*(
                Radial(c, dim) for _ in range(flow_length)
            ))
        elif self.flow_type == 'iaf_flow':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, z):
        sum_log_jacobians = 0
        z = z.repeat(self.c, 1, 1)
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(
            self.mean.repeat(z.size(1), 1, 1).permute(1, 0, 2),
            self.cov.repeat(z.size(1), 1, 1, 1).permute(1, 0, 2, 3)
        ).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x