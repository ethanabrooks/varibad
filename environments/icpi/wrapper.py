import gym


class ArrayWrapper(gym.Wrapper):
    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.obs_array()
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        return self.env.obs_array()
