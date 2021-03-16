import time
import gym
import highway_env
from rl_agents.agents.common.factory import agent_factory

env = gym.make('highway-v0')
# env = gym.make("merge-v0")
env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)

# env.configure({
#     "manual_control": True
# })
done = False
state = env.reset()

while not done:
    env.render()

    # take a random action
    randomAction = env.action_space.sample()
    observation, reward, done, info = env.step(randomAction) 
    # env.step(env.action_space.sample())

    # print(observation)
    print(info)
    # time.sleep(0.2)

env.close()