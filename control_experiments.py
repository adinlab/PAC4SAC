
import time, argparse, torch
import numpy as np
from make_environment import Experiment
parser = argparse.ArgumentParser()
args = parser.parse_args()

class ControlExperiment(Experiment):
    def __init__(self):
        super(ControlExperiment, self).__init__()
        self.device_str = "cpu"
        self.optimizer_args = {"lr": 4e-3}
        self.n_total_steps = 0
        self.max_steps = 100000
        self.eval_frequency=1000
        self.eval_episodes=10
        self.gamma=0.99
        self.warmup_steps=10000
        self.learn_frequency=1
        self.max_iter=1
        self.env = self.env
    
    def train(self):
        time_start = time.time()

        information_dict = {
            "episode_rewards": np.zeros(10000),
            "episode_steps": np.zeros(10000),
            "step_rewards": np.empty((2 * 100000), dtype=object)
        }

        s, _ = self.env.reset()
        r_cum = 0
        episode = 0
        e_step = 0

        for step in range(self.max_steps):

            e_step +=1

            if step % self.eval_frequency == 0:
                self.eval(step)

            if step < self.warmup_steps:
                a = self.env.action_space.sample()
            else:
                a = self.agent.select_action(s)

            a = np.clip(a,-1.0,1.0)
            sp, r, done, truncated, info = self.env.step(a)
            self.agent.store_transition(s,a,r,sp,done,step+1)
            information_dict["step_rewards"][self.n_total_steps + step] = (episode, step, r)

            s = sp # Update state
            r_cum += r # Update cumulative reward

            if step >= self.warmup_steps and (step % self.learn_frequency) == 0:
                self.agent.learn(max_iter=self.max_iter)

            if done or truncated:

                information_dict["episode_rewards"][episode] = r_cum
                information_dict["episode_steps"][episode] = step
                print('Episode:', episode, ' Reward: %.3f' % np.mean(r_cum), 'N-steps: %d' % step)
                s, _ = self.env.reset()
                r_cum = 0
                episode += 1
                e_step = 0

        self.eval(step)
        time_end = time.time()


    @torch.no_grad()
    def eval(self, n_step):
        self.agent.eval()
        results = np.zeros(self.eval_episodes)
        q_values = np.zeros(self.eval_episodes)
        avg_reward=np.zeros(self.eval_episodes)

        for episode in range(self.eval_episodes):
            s, _ = self.eval_env.reset()
            step = 0
            a = self.agent.select_action(s, is_training=False)
            q_values[episode] = self.agent.Q_value(s,a)
            done = False

            while not done:
                a = self.agent.select_action(s, is_training=False)
                sp, r, term, trunc, info = self.eval_env.step(a)
                done = term or trunc
                s = sp
                results[episode] += r
                avg_reward[episode] += self.gamma**step * r
                step += 1

        self.agent.train()