from collections import OrderedDict
from collections import deque
import inspect
import sys

import numpy as np
from gym import spaces

import time
from pickle import dumps,loads
import cloudpickle
from collections import namedtuple

from . import VecEnv


class SnapshotVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """

    def __init__(self, env_fns, snapshot_save_prob=0, snapshot_load_prob=0, verbose=1, visualize=False, human_snapshots=False, training_len=0):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space
        print(obs_space)

        if isinstance(obs_space, spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
        else:
            subspaces = {None: obs_space}

        for key, box in subspaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

        self.is_env_atari = "AtariEnv" in str(self.envs[0].unwrapped)
        self.snapshot_save_prob = snapshot_save_prob
        self.snapshot_load_prob = snapshot_load_prob
        self.first_time = True
        self.verbose = verbose
        self.visualize = visualize

        self.snapshot_buffer = deque(maxlen=10)
        self.snapshot_buffer.append(self.get_snapshot())

        #Human snapshot part
        self.human_snapshots = human_snapshots
        self.training_len = training_len
        self.num_snapshots = 30
        self.timestep = 0
        if human_snapshots:
            print("Loading Snapshots from human demonstration.")
            self.load_snapshots_from_folder()
    # From https://github.com/openai/gym/pull/575


    def load_snapshots_from_folder(self, path="/home/batu/Desktop/TrainingCamp/stable-baseline/HumanSnapshots/MountainCar", num_snapshots=30):
        self.snapshot_buffer = []
        # self.snapshot_buffer.append(self.get_snapshot())
        for i in range(num_snapshots):
            snapshot = cloudpickle.load(open(f"{path}/{i+1}.p", "rb"))
            self.snapshot_buffer.append(snapshot)
        self.snapshot_buffer.reverse()

    def save_snapshot(self, env_id=0):
        # start_time = time.time()
        snapshot = self.get_snapshot(env_id)
        self.snapshot_buffer.append(snapshot)
        if self.verbose:
            print("Saved.")

    def get_snapshot(self, env_id=0):
        """
       :returns: environment state that can be loaded with load_snapshot
       Snapshots guarantee same env behaviour each time they are loaded.

       Warning! Snapshots can be arbitrary things (strings, integers, json, tuples)
       Don't count on them being pickle strings when implementing MCTS.

       Developer Note: Make sure the object you return will not be affected by
       anything that happens to the environment after it's saved.
       You shouldn't, for example, return selfs.env.
       In case of doubt, use pickle.dumps or deepcopy.

       """
        # print("Get snapshot is called.")
        #self.envs[0].render() #close popup windows since we can't pickle them
        #self.envs[env_id].close()
        return cloudpickle.dumps(self.envs[env_id])

    def load_snapshot(self,snapshot, env_id=0):
        """
       Loads snapshot as current env state.
       Should not change snapshot inplace (in case of doubt, deepcopy).
       """

        assert not hasattr(self,"_monitor") or hasattr(self.envs[0].env,"_monitor"), "can't backtrack while recording"

        # print("Load snapshot is called.")
        #self.envs[0].render() #close popup windows since we can't load into them
        # self.envs[env_id].close()
        # start_time = time.time()
        self.envs[env_id].env = loads(snapshot)
        # print (time.time() - start_time, 's in LOAD')

        #if self.verbose:
            #print("Loaded.")


    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.timestep += 1
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                self.envs[env_idx].step(self.actions[env_idx])

            if self.visualize:
                self.envs[env_idx].render()

            if (np.random.random() < self.snapshot_save_prob) and (not self.buf_dones[env_idx]):
                self.save_snapshot(env_idx)
                #self.record_checkpoint(obs)

            if self.buf_dones[env_idx]:
                # This code is for Physics based environments
                # If we can get the state and the load prob kicked in
                if self.is_env_atari:
                    # Human demonstratin snapshot.
                    if (not self.first_time and np.random.random() < self.snapshot_load_prob):
                        self.first_time = False
                        index = np.random.choice(range(len(list(self.snapshot_buffer))))
                        snapshot = self.snapshot_buffer[index]
                        # print("SIZE:")
                        # print(sys.getsizeof(snapshot))
                        self.load_snapshot(snapshot, env_idx)
                        # print(f"Snapshot index {index} was chosen with state \n")

                        # Take a noop action
                        # This is not ideal but trying out as a shorthand.
                        #print(self.envs[env_idx].env.env.env.env.env)
                        # Reaching into monitor.
                        # Make these into cleaner loops.
                        self.envs[env_idx].env.env.env.env.env.needs_reset = False;
                        self.envs[env_idx].env.env.env.env.env.rewards = []

                        #Reaching into TimeLimit
                        self.envs[env_idx].env.env.env.env.env.env.env.env._episode_started_at = time.time()
                        self.envs[env_idx].env.env.env.env.env.env.env.env._elapsed_steps = 0

                        obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                            self.envs[env_idx].step(0)

                        if type(obs) == type(None):
                            obs = self.envs[env_idx].reset()
                    else:
                        self.first_time = False
                        obs = self.envs[env_idx].reset()
                #If it is not atari.
                else:
                    if self.human_snapshots and (not self.first_time) and (np.random.random() < self.snapshot_load_prob):
                        self.first_time = False
                        index = (self.timestep / self.training_len) * self.num_snapshots
                        index = int(index)
                        snapshot = self.snapshot_buffer[index]
                        self.envs[env_idx].unwrapped.state = snapshot
                        obs = snapshot

                        self.envs[env_idx].env._episode_started_at = time.time()
                        self.envs[env_idx].env._elapsed_steps = 0

                        self.envs[env_idx].needs_reset = False;
                        self.envs[env_idx].rewards = []

                        # print(f"Snapshot index {index} was chosen with state \n{obs}")
                        obs = self.envs[env_idx].unwrapped.state
                        if type(obs) == type(None):
                            print("OBS IS ZERO")
                            obs = self.envs[env_idx].reset()

                    elif (not self.first_time and np.random.random() < self.snapshot_load_prob):
                        self.first_time = False
                        index = np.random.choice(range(len(list(self.snapshot_buffer))))
                        snapshot = self.snapshot_buffer[index]
                        self.load_snapshot(snapshot, env_idx)
                        self.envs[env_idx].env._episode_started_at = time.time()
                        self.envs[env_idx].env._elapsed_steps = 0

                        # print(f"Snapshot index {index} was chosen with state \n{obs}")
                        obs = self.envs[env_idx].unwrapped.state
                        if type(obs) == type(None):
                            obs = self.envs[env_idx].reset()
                    else:
                        self.first_time = False
                        obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (np.copy(self._obs_from_buf()), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def record_checkpoint(self, obs):
        with open("mountaincar_sh_data_2.txt", 'a+') as file:
            x1 = str(obs[0])
            x2 = str(obs[1])
            file.write(f"{x1},{x2}\n")

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            print(self.envs[0])
            self._save_obs(env_idx, obs)
        return np.copy(self._obs_from_buf())

    def close(self):
        return

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, *args, **kwargs):
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        if self.keys == [None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs
