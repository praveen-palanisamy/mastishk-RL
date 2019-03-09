from gym.envs.registration import register

register(
    id='Advection-AdG-v0',
    entry_point='gym_advection.envs:AdvectionEnv',
)

register(
	id='Advection-AdG-v1',
	entry_point='gym_advection.envs:AdvectionEnvContinuous',
)
