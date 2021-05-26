from gym.envs.registration import register

register(
    id='cartpole-half-gravity-v0',
    entry_point='custom_gym.envs:CartPoleHalfEnv',
)
register(
    id='cartpole-quarter-gravity-v0',
    entry_point='custom_gym.envs:CartPoleQuaterEnv',
)
register(
    id='cartpole-swing-up-v0',
    entry_point='custom_gym.envs:CartPoleSwingUpEnv',
)
