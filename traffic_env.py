import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficLightEnv(gym.Env):
    """
    Custom Gym environment for traffic light control.
    State: vehicle counts for 4 directions [north, south, east, west]
    Action: green time duration (discrete: 10-60 seconds in 5s increments)
    Reward: negative of average wait time across all vehicles
    """
    def __init__(self):
        super(TrafficLightEnv, self).__init__()

        # State space: 4 vehicle counts (0-50 each)
        self.observation_space = spaces.MultiDiscrete([51, 51, 51, 51])

        # Action space: green time from 10 to 60 seconds in 5s steps (11 actions)
        self.action_space = spaces.Discrete(11)

        # Initialize state
        self.state = None
        self.wait_times = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        self.reset()

    def reset(self, seed=None, options=None):
        # Random initial vehicle counts
        self.state = np.random.randint(0, 51, 4)
        self.wait_times = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        return self.state, {}

    def step(self, action):
        # Convert action to green time (10 + action*5 seconds)
        green_time = 10 + action * 5

        # Simulate traffic flow during green time
        directions = ['North', 'South', 'East', 'West']
        active_directions = ['North', 'South'] if np.random.choice([0, 1]) == 0 else ['East', 'West']

        # Vehicles leaving during green time
        vehicles_served = {}
        for dir in directions:
            if dir in active_directions:
                # Serve vehicles proportional to green time
                served = min(self.state[directions.index(dir)], int(green_time / 2))
                self.state[directions.index(dir)] -= served
                vehicles_served[dir] = served
            else:
                vehicles_served[dir] = 0

        # Update wait times
        for dir in directions:
            if dir not in active_directions:
                # Waiting vehicles accumulate wait time
                self.wait_times[dir] += green_time
            else:
                # Served vehicles reset wait time
                self.wait_times[dir] = 0

        # Add new vehicles (simulate arrival)
        new_vehicles = np.random.randint(0, 10, 4)
        self.state = np.clip(self.state + new_vehicles, 0, 50)

        # Calculate reward: negative average wait time
        total_wait = sum(self.wait_times.values())
        total_vehicles = sum(self.state)
        avg_wait = total_wait / max(total_vehicles, 1)
        reward = -avg_wait

        # Episode never ends in this simple version
        done = False

        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        directions = ['North', 'South', 'East', 'West']
        print(f"Vehicle counts: {dict(zip(directions, self.state))}")
        print(f"Wait times: {self.wait_times}")
