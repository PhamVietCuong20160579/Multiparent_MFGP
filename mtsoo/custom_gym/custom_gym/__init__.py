from gym.envs.registration import register

register(
    id='cartpole-half-gravity-v0',
    entry_point='custom_gym.envs:CartPoleHalfEnv',
)
register(
    id='cartpole-14mm-v0',
    entry_point='custom_gym.envs:CartPole7',
)
register(
    id='cartpole-20mm-v0',
    entry_point='custom_gym.envs:CartPole10',
)
register(
    id='doublepole-710-v0',
    entry_point='custom_gym.envs:DoublePole710',
)
register(
    id='doublepole-510-v0',
    entry_point='custom_gym.envs:DoublePole510',
)
