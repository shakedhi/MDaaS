from gym.envs.registration import register

register(
    id='MDaaS-v1',
    entry_point='mdaas.envs:MdaasEnv',
)
