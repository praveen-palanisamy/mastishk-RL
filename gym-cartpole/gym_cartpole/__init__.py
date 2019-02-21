from gym.envs.registration import register

register(
    id='CartPole-AdG-v0',
    entry_point='gym_cartpole.envs:CartPoleEnv',
)
