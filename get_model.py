from architectures import ActorNetProbabilistic
from pac4sac import PAC4SAC

def get_model( env):
    model_name = 'pac4sac'
    actor_nn = ActorNetProbabilistic
    return PAC4SAC(env)

