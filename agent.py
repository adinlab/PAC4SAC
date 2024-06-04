
import torch.nn as nn
import numpy as np
import torch,argparse
from model_basic import Actor,CriticEnsemble
from experience_memory import ExperienceMemory
parser = argparse.ArgumentParser()
args = parser.parse_args()
args, unknown = parser.parse_known_args()

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        #self.args = args
        self.device = 'cpu'
        self.env = env
        args.buffer_size=1000000
        self._nx, self._nu = self.env.observation_space.shape, self.env.action_space.shape
        print(self._nx)
        print(self._nu)
        self._nx_flat, self._nu_flat = np.prod(self._nx), np.prod(self._nu)
        self._u_min = torch.from_numpy(self.env.action_space.low).float().to(self.device)
        self._u_max = torch.from_numpy(self.env.action_space.high).float().to(self.device)
        self._x_min = torch.from_numpy(self.env.observation_space.low).float().to(self.device)
        self._x_max = torch.from_numpy(self.env.observation_space.high).float().to(self.device)

        self._gamma = 0.99
        self._tau = 0.005

        args.dims = {
            "state": (args.buffer_size, self._nx_flat),
            "action": (args.buffer_size, self._nu_flat),
            "next_state": (args.buffer_size, self._nx_flat),
            "reward": (args.buffer_size),
            "terminated": (args.buffer_size),
            "step": (args.buffer_size)
        }

        self.experience_memory = ExperienceMemory(args)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self._tau*local_param.data + (1.0-self._tau)*target_param.data)

    def _hard_update(self, local_model, target_model):


        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def learn(self, max_iter=1):
        raise NotImplementedError(f"learn() not implemented for {self.name} agent")

    def select_action(self, warmup=False, exploit=False):
        raise NotImplementedError(f"select_action() not implemented for {self.name} agent")

    def store_transition(self, s, a, r, sp, terminated, step):
        self.experience_memory.add(s, a, r, sp, terminated, step)
    
########################
class ActorCritic(Agent):
    _agent_name = "AC"

    def __init__(
        self,
        env,
        actor_nn,
        critic_nn,
        CriticEnsembleType=CriticEnsemble,
        ActorType=Actor,
        policy_delay=1,
    ):
        super(ActorCritic, self).__init__(env)
        self.critics = CriticEnsembleType(critic_nn, self._nx, self._nu)
        self.actor = ActorType(actor_nn,  self._nx, self._nu)
        self.n_iter = 0
        self.policy_delay = policy_delay
        self.batch_size =256

    def learn(self, max_iter=1):
        if self.batch_size > len(self.experience_memory):
            return None

        for iteration in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.batch_size
            )
            y = self.critics.get_bellman_target(r, sp, done, self.actor)
            self.critics.update(s, a, y)
            
            if self.n_iter % self.policy_delay == 0:
                self.actor.update(s, self.critics)
            self.critics.update_target()

            self.n_iter += 1

    @torch.no_grad()
    def select_action(self, s, is_training=True):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        a, _ = self.actor.act(s, is_training=is_training)
        a = a.cpu().numpy().squeeze(0)
        return a

    def Q_value(self, s, a):
        s = torch.from_numpy(s).view(1, -1).float().to(self.device)
        a = torch.from_numpy(a).view(1, -1).float().to(self.device)
        q = self.critics[0].Q(s, a)
        return q.item()