import torch,math
from torch import nn
from torch.nn.modules.loss import _Loss
from architectures import calculate_kl_terms

class Actor(nn.Module):
    def __init__(self, arch,  n_state, n_action):
        super(Actor, self).__init__()
        self.n_hidden = 256
        self.device ="cpu"
        self.learning_rate = 3e-4
        self.model = arch(n_state, n_action, self.n_hidden).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)


    def act(self, s, is_training=True):
        a, e = self.model(
            s, is_training=is_training
        )  
        return a, e

    def loss(self, s, a, e, critics, alpha):
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)
        return (-q + alpha * e).mean()

    def update(self, s, critics, alpha):
        self.optim.zero_grad()
        a, e = self.act(s)
        loss = self.loss(s, a, e, critics, alpha)
        loss.backward()
        self.optim.step()
        return a, e
    

class Critic(nn.Module):
    def __init__(self, arch,  n_state, n_action):
        super(Critic, self).__init__()
        # self.args = args
        self.n_hidden = 256
        self.device = "cpu"
        self.learning_rate = 3e-4
        self.tau = 0.005
        self.model = arch(n_state, n_action, self.n_hidden).to(self.device)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.target = arch(n_state, n_action, self.n_hidden).to(self.device)
        self.init_target()


    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data
                + (1.0 - self.tau) * target_param.data
            )

    def Q(self, s, a):
        return self.model(s, a)

    def Q_t(self, s, a):
        return self.target(s, a)

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), y)
        loss.backward()
        self.optim.step()
        
class CriticEnsemble(nn.Module):
    def __init__(self, arch,  n_state, n_action, critictype=Critic):
        super(CriticEnsemble, self).__init__()
        self.n_elements = 2
        self.gamma = 0.99
        #self.args = args
        self.critics = [
            critictype(arch,  n_state, n_action) for _ in range(self.n_elements)
        ]
        

    def __getitem__(self, item):
        return self.critics[item]

    def Q(self, s, a):
        return [critic.Q(s, a) for critic in self.critics]

    def Q_t(self, s, a):
        return [critic.Q_t(s, a) for critic in self.critics]

    def update(self, s, a, y):
        [critic.update(s, a, y) for critic in self.critics]

    def update_target(self):
        [critic.update_target() for critic in self.critics]

    def reduce(self, q_val_list):
        # Reduces the outputs of ensemble elements into a single value
        return torch.cat(q_val_list, dim=-1).min(dim=-1, keepdim=True)[0]

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor, alpha):
        ap, ep = actor.act(sp)
        qp = self.Q_t(sp, ap)
        qp_t = self.reduce(qp) - alpha * ep
        y = r.unsqueeze(-1) + (self.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y
####################################################
class NormalMLELoss(_Loss):

    def forward(self, mu, logvar, y):
        logvar = logvar.clamp(-4.6, 4.6) # log(-4.6) = 0.01
        var = logvar.exp()
        return (0.5*logvar + 0.5*((mu - y).pow(2)) / var).mean()
#####################################################################
# Analytical loss for a last-layer BNN
class NormalLLLoss(_Loss):
    def forward(self, mu, lvar, y):
        var = lvar.clamp(-4.6, 4.6).exp()
        return ((mu - y).pow(2) + var).mean()
#####################################################################
# Approx McAllester Bound (as used in the PAC4SAC draft)
# TODO: Generalize this to a general McAllester bound
class McAllester(_Loss):
    def forward(self, critic, N, delta=0.05):
        confidence_term = math.log(2.0*math.sqrt(N)/delta)
        return ((calculate_kl_terms(critic)[0] + confidence_term) / (2*N)).sqrt().mean()


#####################################################################
class ProbCritic(Critic):
    def __init__(self, arch,  n_state, n_action):
        super(ProbCritic, self).__init__(arch,  n_state, n_action)
        self.loss = NormalMLELoss()

    def Q_t(self, s, a):
        return self.target(s, a)[:, 0].view(-1, 1)

    def Q(self, s, a):
        return self.model(s, a)[:, 0].view(-1, 1)

    def get_distribution(self, s, a, is_target=False):
        if is_target:
            out = self.target(s, a)
        else:
            out = self.model(s, a)
        mu = out[:, 0].view(-1, 1)
        logvar = out[:, 1].view(-1, 1)
        return mu, logvar

    def Q_dist(self, s, a):
        return self.get_distribution(s, a)

    def Q_t_dist(self, s, a):
        return self.get_distribution(s, a, is_target=True)

    def update(self, s, a, y):
        self.optim.zero_grad()
        mu, logvar = self.get_distribution(s, a)
        self.loss(mu, logvar, y).backward()
        self.optim.step()


#####################################################################
class ProbCriticEnsemble(CriticEnsemble):
    def __init__(self, arch,  n_state, n_action, CriticType=ProbCritic):
        super(ProbCriticEnsemble, self).__init__(
            arch,  n_state, n_action, CriticType
        )
        self.gamma = 0.99

    def get_reduced_distribution(self, s, a, is_target=False):
        if is_target:
            val = [critic.target(s, a) for critic in self.critics]
        else:
            val = [critic.model(s, a) for critic in self.critics]
        val = torch.cat(val, dim=-1)
        mu_list, var_list = val[:, 0::2], val[:, 1::2].clamp(-4.6, 4.6).exp()
        idx = mu_list.argmin(dim=-1)
        mu_e = mu_list.gather(1, idx.unsqueeze(-1))
        var_e = var_list.gather(1, idx.unsqueeze(-1))
        var_a = torch.zeros_like(var_e)
        return mu_e, var_e, var_a

    # First two moments of a mixture of two unimodal distributions
    def Q_dist(self, s, a):
        return self.get_reduced_distribution(s, a, is_target=False)

    def Q_t_dist(self, s, a):
        return self.get_reduced_distribution(s, a, is_target=True)
    
    def Q(self, s, a):
        mu, var_a, var_e = self.get_reduced_distribution(s, a, is_target=False)
        return mu + (var_a+var_e).sqrt()*torch.randn_like(mu)
    
    def Q_t(self, s, a):
        mu, var_a, var_e = self.get_reduced_distribution(s, a, is_target=True)
        return mu + (var_a+var_e).sqrt()*torch.randn_like(mu)

    @torch.no_grad
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        if ep is None:
            ep = 0
        mu, var_e, var_a = self.Q_t_dist(sp, ap)
        qp_t = mu - alpha * ep
        y = r.unsqueeze(-1) + (self.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y
    




