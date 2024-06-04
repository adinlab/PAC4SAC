import math,torch
from agent import ActorCritic
from architectures import ActorNetProbabilistic,CriticNet
from model_basic import Actor,CriticEnsemble


class SoftActor(Actor):
    def __init__(self, arch, n_state, n_action, has_target=False):
        super(SoftActor, self).__init__(arch, n_state, n_action)
        self.alpha=0.2
        self.n_hidden = 256
        self.learning_rate = 3e-4
        self._step = 0
        self._max_steps = 100000
        self.batch_size=256
        self._gamma=0.99
        self.H_target = -n_action[0]
        self.log_alpha = torch.tensor(
            math.log(self.alpha), requires_grad=True, device=self.device
        )
        self.optim_alpha = torch.optim.Adam([self.log_alpha], self.learning_rate)

    # TODO: Consider updating to the version that Abdullah uses
    def update_alpha(self, e):
        self.optim_alpha.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (e + self.H_target).detach()).mean()
        alpha_loss.backward()
        self.optim_alpha.step()

    def loss(self, s, critics):
        a, e = self.act(s)
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)
        return (-q + self.log_alpha.exp() * e).mean(), e

    def update(self, s, critics):
        self.optim.zero_grad()
        loss, e = self.loss(s, critics)
        loss.backward()
        self.optim.step()
        self.update_alpha(e)


#####################################################################
class SoftActorCritic(ActorCritic):
    _agent_name = "SAC"

    def __init__(
        self,
        env,
        
        actor_nn=ActorNetProbabilistic,
        critic_nn=CriticNet,
        CriticEnsembleType=CriticEnsemble,
        ActorType=SoftActor,
    ):
        super(SoftActorCritic, self).__init__(
            env,  actor_nn, critic_nn, CriticEnsembleType, ActorType
        )
        self.alpha=0.2
        self.n_hidden = 256
        self.learning_rate = 3e-4
        self._step = 0
        self._max_steps = 100000
        self.batch_size=256
        self._gamma=0.99
