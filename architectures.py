
import torch,math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

####################
def calculate_kl_terms(model: nn.Module):
    """Function to calculate KL loss of bayesian neural network"""
    kl, n = 0, int(0)
    for m in model.modules():
        if m.__class__.__name__.startswith("Variational"):
            kl_, n_ = m.kl_loss()
            kl += kl_
            n += n_
        if m.__class__.__name__.startswith("CLTLayer"):
            kl_, n_ = m.KL()
            kl += kl_
            n += n_
    return kl, n
#########################
class VariationalBayesianLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, prior_log_sig2=0
    ) -> None:
        super(VariationalBayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_mu_prior = nn.Parameter(
            torch.ones((out_features, in_features)), requires_grad=False
        )
        self.weight_log_sig2_prior = nn.Parameter(
            prior_log_sig2 * torch.ones((out_features, in_features)),
            requires_grad=False,
        )
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_mu", None)
        self.reset_parameters()

    def reset_parameters(self, ) -> None:
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, -10)
        if self.has_bias:
            init.zeros_(self.bias_mu)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(
            input.pow(2), self.weight_log_sig2.clamp(-10,10).exp(), bias=None
        )
        
        return output_mu + output_sig2.sqrt() * torch.randn_like(
            output_sig2
        )

    def get_mean_var(self, input: torch.Tensor) -> torch.Tensor:
        mu = F.linear(input, self.weight_mu, self.bias_mu)
        sig2 = F.linear(
            input.pow(2), self.weight_log_sig2.clamp(-10,10).exp(), bias=None
        )

        return mu, sig2

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.has_bias
        )

    def update_prior(self, newprior):
        self.weight_mu_prior.data = newprior.weight_mu.data.clone()
        self.weight_mu_prior.data.requires_grad = False
        self.weight_log_sig2_prior.data = newprior.weight_log_sig2.data.clone()
        self.weight_log_sig2_prior.data.requires_grad = False
        #if self.has_bias:
        #    self.bias_mu_prior.data = newprior.bias_mu.data.clone()
        #    self.bias_mu_prior.data.requires_grad = False
        #    self.bias_log_sig2_prior.data = newprior.bias_log_sig2.data.clone()
        #    self.bias_log_sig2_prior.data.requires_grad = False

    def kl_loss(self):
        kl_weight = 0.5 * (
            self.weight_log_sig2_prior
            - self.weight_log_sig2.clamp(-8,8)
            + (
                self.weight_log_sig2.clamp(-8,8).exp()
                + (self.weight_mu_prior - self.weight_mu) ** 2
            )
            / self.weight_log_sig2_prior.exp()
            - 1.0
        )
        kl = kl_weight.sum()
        n = len(self.weight_mu.view(-1))
        #if self.has_bias:
        #    kl_bias = 0.5 * (
        #        self.bias_log_sig2_prior
        #        - self.bias_log_sig2
        #        + (self.bias_log_sig2.exp() + (self.bias_mu_prior - self.bias_mu) ** 2)
        #        / (self.bias_log_sig2_prior.exp())
        #        - 1.0
        #    )
        #    kl += kl_bias.sum()
        #    n += len(self.bias_mu.view(-1))
        return kl, n


class CriticNetEpistemic(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256):
        super(CriticNetEpistemic, self).__init__()
        self.arch = nn.Sequential(
            nn.Linear(n_x[0] + n_u[0], n_hidden),
            nn.ReLU(),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            ###
        )
        self.head = VariationalBayesianLinear(256, 1)

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        f = self.arch(f)
        mu, var = self.head.get_mean_var(f)
        return torch.concat([mu, var.log()], dim=-1)

 
   
 
class SquashedGaussianHead(nn.Module):
    def __init__(self, n, upper_clamp=-2.0):
        super(SquashedGaussianHead, self).__init__()
        self._n = n
        self._upper_clamp = upper_clamp

    def forward(self, x, is_training=True):
        # bt means before tanh
        mean_bt = x[..., : self._n]
        log_var_bt = (x[..., self._n :]).clamp(-10, -self._upper_clamp)  # clamp added
        std_bt = log_var_bt.exp().sqrt()
        dist_bt = Normal(mean_bt, std_bt)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist_bt, transform)
        if is_training:
            y = dist.rsample()
            y_logprob = dist.log_prob(y).sum(dim=-1, keepdim=True)
        else:
            y_samples = dist.rsample((100,))
            y = y_samples.mean(dim=0)
            y_logprob = None

        return y, y_logprob  # dist

class ActorNetProbabilistic(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256, upper_clamp=-2.0):
        super(ActorNetProbabilistic, self).__init__()
        self.n_u = n_u
        self.arch = nn.Sequential(
            nn.Linear(n_x[0], n_hidden),
            nn.ReLU(),
            ##
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            ##
            nn.Linear(n_hidden, 2 * n_u[0]),
        )
        self.head = SquashedGaussianHead(self.n_u[0], upper_clamp)

    def forward(self, x, is_training=True):
        f = self.arch(x)
        return self.head(f, is_training)
    
class CriticNet(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256):
        super(CriticNet, self).__init__()
        self.arch = nn.Sequential(
            nn.Linear(n_x[0] + n_u[0], n_hidden),
            nn.ReLU(),
            ##
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            ###
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        return self.arch(f)