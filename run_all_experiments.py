import os
import gym
import torch
import random
import numpy as np
import collections
import pandas as pd
from stable_baselines3 import SAC, TD3
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from UAVUncertain.BasicEnv70 import SmoothDroneEnv, DualMetaContextWrapper #6
from UAVUncertain.MetaSAC import PrioritizedReplayBuffer, DualMetaSAC
from UAVUncertain.Map0 import generate_shanghai_logistics_map
from UAVUncertain.Without_Couple import PrioritizedReplayBuffer_WC,DualMetaSAC_WC
from UAVUncertain.WihtoutPER import UniformReplayBuffer_WP,DualMetaSAC_WP
from UAVUncertain.PERSAC import PrioritizedReplayBuffer_WG,SAC_PER

SEEDS = [10,30,50,70,90]
buffer_WL=PrioritizedReplayBuffer(10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent_WL = DualMetaSAC(device)

def set_global_seeds_torch(seed=SEEDS[0]):#42

    print(f">>> Setting global random seed (PyTorch): {seed} <<<")

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

shanghai_wp = generate_shanghai_logistics_map()[0]

# Training Codes
# Global Settings and Folder Initialization

EPOCHS = 1000
STEPS_PER_EPOCH = 1000

for folder in ["models", "results", "tb_logs", "logs"]:
    os.makedirs(folder, exist_ok=True)


# Baseline Environment Wrapper

class FairBaselineWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.wind_range, self.mass_var, self.max_delay = 0.0, 0.0, 0
        self.delay_buffer = collections.deque(maxlen=10)
        self.ou_theta, self.ou_sigma = 0.15, 0.2
        self.target_wind = np.zeros(3)
        self.curr_delay = 0

    def _sample_wind(self):
        if self.wind_range <= 1e-5: return np.zeros(3)
        w = np.random.uniform(-self.wind_range, self.wind_range, size=3)
        w[2] *= 0.3
        return w

    def reset(self, random_start=True, **kwargs):
        self.unwrapped.m = self.unwrapped.base_m * np.random.uniform(1.0 - self.mass_var, 1.0 + self.mass_var)
        self.target_wind = self._sample_wind()
        self.unwrapped.wind = self.target_wind.copy()
        self.curr_delay = np.random.randint(0, self.max_delay + 1) if self.max_delay > 0 else 0
        self.delay_buffer.clear()
        for _ in range(self.curr_delay): self.delay_buffer.append(np.zeros(3))

        obs = self.env.reset(random_start=random_start)
        return obs.astype(np.float32)

    def step(self, action):
        if np.random.rand() < 0.05: self.target_wind = self._sample_wind()
        if self.max_delay > 0 and np.random.rand() < 0.01: self.curr_delay = np.random.randint(0, self.max_delay + 1)

        if self.wind_range > 1e-5:
            wind_diff = self.target_wind - self.unwrapped.wind
            noise = np.random.randn(3) * self.ou_sigma * self.wind_range
            noise[2] *= 0.3
            self.unwrapped.wind += self.ou_theta * wind_diff + noise
            self.unwrapped.wind = np.clip(self.unwrapped.wind,
                                          [-self.wind_range, -self.wind_range, -self.wind_range * 0.3],
                                          [self.wind_range, self.wind_range, self.wind_range * 0.3])

        self.delay_buffer.append(action)
        exec_action = self.delay_buffer.popleft() if self.curr_delay > 0 else action

        obs, reward, done, info = self.env.step(exec_action)
        return obs.astype(np.float32), reward, done, info

    def set_curriculum(self, w, m, d):
        self.wind_range, self.mass_var, self.max_delay = w, m, d


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_epoch = -1


    def _on_step(self) -> bool:
        epoch = self.num_timesteps // STEPS_PER_EPOCH

        if epoch != self.last_epoch and epoch % 1 == 0:
            print(f"[SB3 Baseline 进度] Epoch: {epoch}/1000")
            self.last_epoch = epoch
        if epoch < 400:
            w, m, d = 0.0, 0.0, 0
        elif epoch < 600:
            w, m, d = 2.0, 0.1, 1
        elif epoch < 800:
            w, m, d = 4.0, 0.2, 2
        else:
            w, m, d = 6.0, 0.3, 5
        self.training_env.env_method("set_curriculum", w, m, d)
        return True


# Training Module for Our Mode


def evaluate_metasac(env, agent, is_level_0, episodes=1):
    avg_r = 0.
    for _ in range(episodes):
        state = env.reset(random_start = False, determine = True)
        done = False;
        step = 0
        while not done and step < 8000:
            action = agent.take_action(state, deterministic=True)
            state, r, done, _ = env.step(action)
            avg_r += r;
            step += 1
    return avg_r / episodes


def train_metasac(seed):
    set_global_seeds_torch(seed)

    # Environment initialization
    env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    eval_env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    agent = DualMetaSAC(device)
    buffer = PrioritizedReplayBuffer(100000)

    # Course monitoring variables
    current_lesson = 0
    success_window = collections.deque(maxlen=10)
    score_window = collections.deque(maxlen=5)
    eval_returns = []
    avg_score = -np.inf
    best_score = 3800
    # Lesson 0
    w, m, d = 0.0, 0.0, 0
    is_level_0 = True
    successcount = 0
    beta_offset = 0.0

    for epoch in range(EPOCHS):
        # 1. Update environment difficulty parameters
        env.wind_range, env.mass_var, env.max_delay = w, m, d
        eval_env.wind_range, eval_env.mass_var, eval_env.max_delay = w, m, d

        # 2. Calculate the Beta Annealing of PER
        current_beta = 0.4 + (1.0 - 0.4) * min(1.0, epoch / 400.0)

        # 3. Training Loops
        state = env.reset(random_start=True)
        ep_steps = 0
        for _ in range(STEPS_PER_EPOCH):
            action = agent.take_action(state, deterministic=False, is_level_0=is_level_0)
            next_state, reward, done, info = env.step(action)

            # Handle timeout truncation to prevent underestimation of Q value
            real_done = done if info['info_status'] != 'timeout' else False

            buffer.add(state, action, reward, next_state, real_done, info['gt_params'])
            state = next_state
            ep_steps += 1

            if buffer.size() > 2000:
                batch = buffer.sample(256, beta=current_beta)
                abs_errors, critic_loss, pred_loss = agent.update(batch)
                buffer.update_priorities(batch[6], abs_errors)

            if done:
                state = env.reset(random_start=True)
                ep_steps = 0

        # 4. Regular evaluation and course automation switching
        if epoch % 2 == 0:
            # Evaluation function returns score and status
            score, info = evaluate_metasac(eval_env, agent, is_level_0=is_level_0)
            eval_returns.append(score)
            score_window.append(score)

            # Update monitoring window
            is_success = 1 if info['info_status'] == "success" or info['info_status']=="overshoot" else 0
            success_window.append(is_success)

            avg_win_rate = sum(success_window) / len(success_window) if len(success_window) > 0 else 0

            print(f"Epoch {epoch} | Lesson {current_lesson} | Score: {score:.1f} | "
                  f"Success: {info['info_status']} | win_rate: {avg_win_rate:.2f} | Env: W={w}, M={m}, D={d}")

            # --- Logic of course switching lessons ---
            if current_lesson == 0 and len(success_window) == 10:
                if sum(success_window) >= 8 and score > 4400:
                    print(f"🌟 [Level Up] Lesson 0 收敛！进入 Lesson 1 (微风/短时延阶段)")
                    current_lesson = 1
                    beta_offset += 0.15
                    is_level_0 = False
                    w, m, d = 2.0, 0.1, 0
                    success_window.clear()
                    score_window.clear()
                    buffer.decay_priorities(decay_factor=0.3)

            elif current_lesson == 1 and len(success_window) == 10:
                if sum(success_window) >= 8 and score > 4200:
                    print(f"🌟 [Level Up] Lesson 1 收敛！进入 Lesson 2 (中等干扰阶段)")
                    current_lesson = 2
                    w, m, d = 4.0, 0.2, 0
                    beta_offset += 0.15
                    success_window.clear()
                    score_window.clear()
                    buffer.decay_priorities(decay_factor=0.3)

            elif current_lesson == 2 and len(success_window) == 10:
                if sum(success_window) >= 8 and score > 4200:
                    print(f"🌟 [Level Up] Lesson 2 收敛！进入 Lesson 3 (中等干扰阶段)")
                    current_lesson = 3
                    w, m, d = 4.0, 0.2, 2
                    beta_offset += 0.15
                    success_window.clear()
                    score_window.clear()
                    buffer.decay_priorities(decay_factor=0.3)

            elif current_lesson == 3 and len(success_window) == 10:
                if sum(success_window) >= 8 and score > 4200:
                    print(f"🌟 [Level Up] Lesson 3 收敛！进入 Lesson 4 (极端边界阶段)")
                    current_lesson = 4
                    w, m, d = 6.0, 0.3, 2
                    beta_offset += 0.15
                    success_window.clear()
                    score_window.clear()
                    buffer.decay_priorities(decay_factor=0.3)

            elif current_lesson == 4 and len(score_window) == 5:
                if epoch> 400 and sum(success_window) >= 8 and score > best_score:
                    successcount += 1
                    torch.save(agent.actor.state_dict(), f"models/1lesson3_metasac_seed{seed}_actor_best.h5")
                    torch.save(agent.critic.state_dict(), f"models/1lesson3_metasac_seed{seed}_critic_best.h5")
                    torch.save(agent.encoder.state_dict(), f"models/1lesson3_metasac_seed{seed}_encoder_best.h5")
                    best_score = score
                    print(f"savepoint: Epoch {epoch} | score: {score:.2f}"
                          f"successcount:{successcount} ")
    np.save(f"results/Fig1_MetaSAC_seed{seed}.npy", eval_returns)

def train_withoutlesson(seed):
    set_global_seeds_torch(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    eval_env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    agent = agent_WL
    buffer = buffer_WL

    eval_returns = []
    best_maxwind_score = 5000
    best_score = 5000

    for epoch in range(EPOCHS):
        is_level_0 = (epoch < 0)

        current_beta = 0.4 + (1.0 - 0.4) * min(1.0, epoch / 400.0)

        w, m, d = 6.0, 0.3, 5

        env.wind_range, env.mass_var, env.max_delay = w,m,d
        eval_env.wind_range, eval_env.mass_var, eval_env.max_delay = w,m,d

        state = env.reset(random_start=True)
        ep_steps = 0

        for _ in range(STEPS_PER_EPOCH):
            action = agent.take_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done, info['gt_params'])
            state = next_state
            ep_steps += 1

            if buffer.size() > 2000:
                batch = buffer.sample(256, beta=current_beta)
                abs_errors, _, _ = agent.update(batch)
                buffer.update_priorities(batch[6], abs_errors)

            if done or ep_steps >= 8000:
                state = env.reset(random_start=True)
                ep_steps = 0

        if epoch % 1 == 0:
            score = evaluate_metasac(eval_env, agent,is_level_0)
            eval_returns.append(score)
            print(f"MetaSAC | Seed {seed} | Epoch {epoch} | Score: {score:.2f} | Env: W={w}, M={m}, D={d}")
            if score > best_score:
                best_score = score
                torch.save(agent.actor.state_dict(), f"models/WLmetasac_s{seed}_actor.h5")
                torch.save(agent.encoder.state_dict(), f"models/WLmetasac_s{seed}_encoder.h5")
                torch.save(agent.critic.state_dict(), f"models/WLmetasac_s{seed}_critic.h5")

    np.save(f"results/WLFig1_MetaSAC_seed{seed}.npy", eval_returns)

def train_withoutper(seed):
    set_global_seeds_torch(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    eval_env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    agent = DualMetaSAC_WP(device)
    buffer = UniformReplayBuffer_WP(100000)

    eval_returns = []
    best_maxwind_score = 5000
    best_score = 5000

    for epoch in range(EPOCHS):
        is_level_0 = (epoch < 400)

        current_beta = 0.4 + (1.0 - 0.4) * min(1.0, epoch / 400.0)

        if epoch < 400:
            w, m, d = 0.0, 0.0, 0
        elif epoch < 600:
            w, m, d = 2.0, 0.1, 1
        elif epoch < 800:
            w, m, d = 4.0, 0.2, 2

        else:
            w, m, d = 6.0, 0.3, 5

        env.wind_range, env.mass_var, env.max_delay = w, m, d
        eval_env.wind_range, eval_env.mass_var, eval_env.max_delay = w, m, d

        state = env.reset(random_start=True)
        ep_steps = 0

        for _ in range(STEPS_PER_EPOCH):
            action = agent.take_action(state, deterministic=False, is_level_0=is_level_0)
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done, info['gt_params'])
            state = next_state
            ep_steps += 1

            if buffer.size() > 2000:
                batch = buffer.sample(256, beta=current_beta)
                abs_errors, _, _ = agent.update(batch)
                buffer.update_priorities(batch[6], abs_errors)

            if done or ep_steps >= 8000:
                state = env.reset(random_start=True)
                ep_steps = 0

        if epoch % 1 == 0:
            score = evaluate_metasac(eval_env, agent, is_level_0)
            eval_returns.append(score)
            print(f"MetaSAC | Seed {seed} | Epoch {epoch} | Score: {score:.2f} | Env: W={w}, M={m}, D={d}")
            if score > best_score and not is_level_0:
                best_score = score
                torch.save(agent.actor.state_dict(), f"models/WPmetasac_s{seed}_actor.h5")
                torch.save(agent.encoder.state_dict(), f"models/WPmetasac_s{seed}_encoder.h5")
                torch.save(agent.critic.state_dict(), f"models/WPmetasac_s{seed}_critic.h5")
            if score > best_maxwind_score and epoch > 800:
                best_maxwind_score = score
                torch.save(agent.actor.state_dict(), f"models/lesson3_WPmetasac_seed{seed}_actor_best.h5")
                torch.save(agent.critic.state_dict(), f"models/lesson3_WPmetasac_seed{seed}_critic_best.h5")
                torch.save(agent.encoder.state_dict(), f"models/lesson3_WPmetasac_seed{seed}_encoder_best.h5")

    np.save(f"results/WPFig1_MetaSAC_seed{seed}.npy", eval_returns)

def train_withoutcouple(seed):
    set_global_seeds_torch(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    eval_env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    agent = DualMetaSAC_WC(device)
    buffer = PrioritizedReplayBuffer_WC(100000)

    eval_returns = []
    best_maxwind_score = 5000
    best_score = 5000

    for epoch in range(EPOCHS):
        is_level_0 = (epoch < 400)

        current_beta = 0.4 + (1.0 - 0.4) * min(1.0, epoch / 400.0)

        if epoch < 400:
            w, m, d = 0.0, 0.0, 0
        elif epoch < 600:
            w, m, d = 2.0, 0.1, 1
        elif epoch < 800:
            w, m, d = 4.0, 0.2, 2

        else:
            w, m, d = 6.0, 0.3, 5

        env.wind_range, env.mass_var, env.max_delay = w, m, d
        eval_env.wind_range, eval_env.mass_var, eval_env.max_delay = w, m, d

        state = env.reset(random_start=True)
        ep_steps = 0

        for _ in range(STEPS_PER_EPOCH):
            action = agent.take_action(state, deterministic=False, is_level_0=is_level_0)
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done, info['gt_params'])
            state = next_state
            ep_steps += 1

            if buffer.size() > 2000:
                batch = buffer.sample(256, beta=current_beta)
                abs_errors, _, _ = agent.update(batch)
                buffer.update_priorities(batch[6], abs_errors)

            if done or ep_steps >= 8000:
                state = env.reset(random_start=True)
                ep_steps = 0

        if epoch % 1 == 0:
            score = evaluate_metasac(eval_env, agent, is_level_0)
            eval_returns.append(score)
            print(f"MetaSAC | Seed {seed} | Epoch {epoch} | Score: {score:.2f} | Env: W={w}, M={m}, D={d}")
            if score > best_score and not is_level_0:
                best_score = score
                torch.save(agent.actor.state_dict(), f"models/WCmetasac_s{seed}_actor.h5")
                torch.save(agent.encoder.state_dict(), f"models/WCmetasac_s{seed}_encoder.h5")
                torch.save(agent.critic.state_dict(), f"models/WCmetasac_s{seed}_critic.h5")
            if score > best_maxwind_score and epoch > 800:
                best_maxwind_score = score
                torch.save(agent.actor.state_dict(), f"models/lesson3_WCmetasac_seed{seed}_actor_best.h5")
                torch.save(agent.critic.state_dict(), f"models/lesson3_WCmetasac_seed{seed}_critic_best.h5")
                torch.save(agent.encoder.state_dict(), f"models/lesson3_WCmetasac_seed{seed}_encoder_best.h5")

    np.save(f"results/WCFig1_MetaSAC_seed{seed}.npy", eval_returns)

def train_withoutgru(seed):
    set_global_seeds_torch(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    eval_env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    agent = SAC_PER(device)
    buffer = PrioritizedReplayBuffer_WG(100000)

    eval_returns = []
    best_maxwind_score = 5000
    best_score = 5000

    for epoch in range(EPOCHS):
        is_level_0 = (epoch < 400)

        current_beta = 0.4 + (1.0 - 0.4) * min(1.0, epoch / 400.0)

        if epoch < 400:
            w, m, d = 0.0, 0.0, 0
        elif epoch < 600:
            w, m, d = 2.0, 0.1, 1
        elif epoch < 800:
            w, m, d = 4.0, 0.2, 2

        else:
            w, m, d = 6.0, 0.3, 5

        env.wind_range, env.mass_var, env.max_delay = w, m, d
        eval_env.wind_range, eval_env.mass_var, eval_env.max_delay = w, m, d

        state = env.reset(random_start=True)
        ep_steps = 0

        for _ in range(STEPS_PER_EPOCH):
            action = agent.take_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done, info['gt_params'])
            state = next_state
            ep_steps += 1

            if buffer.size() > 2000:
                batch = buffer.sample(256, beta=current_beta)
                abs_errors, _, _ = agent.update(batch)
                buffer.update_priorities(batch[6], abs_errors)

            if done or ep_steps >= 8000:
                state = env.reset(random_start=True)
                ep_steps = 0

        if epoch % 1 == 0:
            score = evaluate_metasac(eval_env, agent, is_level_0)
            eval_returns.append(score)
            print(f"MetaSAC | Seed {seed} | Epoch {epoch} | Score: {score:.2f} | Env: W={w}, M={m}, D={d}")
            if score > best_score and not is_level_0:
                best_score = score
                torch.save(agent.actor.state_dict(), f"models/WGmetasac_s{seed}_actor.h5")
                torch.save(agent.encoder.state_dict(), f"models/WGmetasac_s{seed}_encoder.h5")
                torch.save(agent.critic.state_dict(), f"models/WGmetasac_s{seed}_critic.h5")
            if score > best_maxwind_score and epoch > 800:
                best_maxwind_score = score
                torch.save(agent.actor.state_dict(), f"models/lesson3_WGmetasac_seed{seed}_actor_best.h5")
                torch.save(agent.critic.state_dict(), f"models/lesson3_WGmetasac_seed{seed}_critic_best.h5")
                torch.save(agent.encoder.state_dict(), f"models/lesson3_WGmetasac_seed{seed}_encoder_best.h5")

    np.save(f"results/WGFig1_MetaSAC_seed{seed}.npy", eval_returns)

import time

def test_model_and_save_data(algo_name, seed):
    """Perform random testing, recovery testing, and special trajectory recording"""
    # Models
    if algo_name == "MetaSAC":
        env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
        agent = DualMetaSAC(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        agent.actor.load_state_dict(torch.load(f"models/metasac_s{seed}_actor.h5"))
        agent.encoder.load_state_dict(torch.load(f"models/metasac_s{seed}_encoder.h5"))
        agent.critic.load_state_dict(torch.load(f"models/metasac_s{seed}_critic.h5"))
        get_action = lambda s: agent.take_action(s, deterministic=True, is_level_0=False)

    # --- Record trajectory ---
    if seed == SEEDS[0]:
        for case, (w, m, d, offset) in {"Fig2_Normal": (3.0, 0.1, 2, 0.0),
                                        "Fig5_Case1": (6.0, 0.3, 0, 0.0),
                                        "Fig5_Case2": (0.0, 0.0, 5, 0.0),
                                        "Fig5_Case3": (6.0, 0.3, 5, 15.0)}.items():
            state = env.reset(random_start = False, determine = True)
            env.wind_range, env.mass_var, env.max_delay = w, m, d
            if offset > 0: env.unwrapped.state[0:2] += np.array([10.0, -offset])

            traj = {'x': [], 'y': [], 'ay': [], 'ref_x': env.unwrapped.dense_path[:, 0],
                    'ref_y': env.unwrapped.dense_path[:, 1]}
            done = False
            while not done:
                # Case 2
                if case == "Fig5_Case2" and np.random.rand() < 0.05: env.curr_delay = np.random.randint(0, 6)
                a = get_action(state)
                state, _, done, _ = env.step(a)
                traj['x'].append(env.unwrapped.state[0]);
                traj['y'].append(env.unwrapped.state[1]);
                traj['ay'].append(a[1])
            np.save(f"results/{case}_{algo_name}_s{seed}.npy", traj)

    # --- Interpretable data ---
    if algo_name == "MetaSAC" and seed == SEEDS[0]:
        log = collections.defaultdict(list)
        state = env.reset(random_start=False, determine = True)
        env.wind_range, env.mass_var, env.max_delay = 6.0, 0.3, 0
        done = False
        while not done:
            a = get_action(state)
            with torch.no_grad():
                meta_obs = torch.tensor([state], dtype=torch.float).to(agent.device)
                obs, s_hist, a_hist = agent._split_obs(meta_obs)
                z = agent.encoder(s_hist, a_hist)
                pred_y = agent.encoder.predictor(z)[0, 1].item() * 6.0
                q1, q2 = agent.critic(obs, z, torch.tensor([a], dtype=torch.float).to(agent.device))

            log['true_wind_y'].append(env.unwrapped.wind[1])
            log['pred_wind_y'].append(pred_y)
            log['z_norm'].append(torch.norm(z).item())
            log['q_value'].append(torch.min(q1, q2).item())
            log['action_y'].append(a[1])
            state, _, done, _ = env.step(a)
        np.save(f"results/MetaSACFig4_Interpretability_{seed}.npy", dict(log))

        # --- Table 1 and Table 2---
        for mode, (w, m, d, offset) in {"Table3": (3.0, 0.15, 1, 0.0), "Table6": (6.0, 0.3, 5, 15.0)}.items():
            metrics = collections.defaultdict(list)

            for _ in range(1000):  #
                state = env.reset(random_start=False, determine=True)
                env.wind_range, env.mass_var, env.max_delay = w, m, d
                if offset > 0: env.unwrapped.state[0:2] += np.array([10.0, -offset])

                done = False
                ep_len = 0
                ep_dist = 0.0
                ep_cte = []
                acts = []
                is_rec = False
                rec_time = 0
                rec_dist = 0.0
                ep_inference_times = []

                while not done and ep_len < 8000:

                    start_time = time.perf_counter()

                    a = get_action(state)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    end_time = time.perf_counter()

                    # Record single step time consumption (ms）
                    ep_inference_times.append((end_time - start_time) * 1000.0)
                    # ==========================================

                    acts.append(a)
                    prev_pos = env.unwrapped.state[0:3].copy()
                    state, _, done, info = env.step(a)
                    curr_pos = env.unwrapped.state[0:3].copy()

                    dist = np.linalg.norm(curr_pos - prev_pos)
                    ep_dist += dist
                    ep_len += 1
                    cte, _ = env.unwrapped._get_closest_point(curr_pos)
                    ep_cte.append(cte)

                    if offset > 0 and not is_rec:
                        rec_time += 1
                        rec_dist += dist
                        if cte < 2.0: is_rec = True

                # Summary of data indicators
                metrics['success'].append(1 if info['info_status'] == 'success' else 0)
                metrics['time'].append(ep_len * 0.1)
                metrics['dist'].append(ep_dist)
                metrics['offset'].append(np.mean(ep_cte))
                metrics['jitter'].append(np.std(np.diff(acts, axis=0)) if len(acts) > 1 else 0)
                metrics['inference_time_ms'].append(np.mean(ep_inference_times) if len(ep_inference_times) > 0 else 0.0)

                if offset > 0:
                    metrics['rec_success'].append(1 if is_rec else 0)
                    metrics['rec_time'].append(rec_time * 0.1 if is_rec else np.nan)
                    metrics['rec_dist'].append(rec_dist if is_rec else np.nan)

            np.save(f"results/{mode}_{algo_name}_seed{seed}.npy", dict(metrics))


def evaluate_robustness(env, agent, wind, delay, mass_var=0.2, eval_episodes=20):

    env.wind_range = wind
    env.max_delay = delay
    env.mass_var = mass_var

    success_count = 0
    for _ in range(eval_episodes):
        state = env.reset(random_start=True)
        done = False
        ep_step = 0
        while not done and ep_step < 8000:
            # Turn off random exploration during data indicator summary
            # testing and rely on Z-vector for feedforward compensation
            action = agent.take_action(state, deterministic=True, is_level_0=False)
            state, _, done, info = env.step(action)
            ep_step += 1

        if info['info_status'] == 'success':
            success_count += 1

    return success_count / eval_episodes


def collect_ood_and_boundary_data(seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DualMetaContextWrapper(SmoothDroneEnv(shanghai_wp))
    agent = DualMetaSAC(device)

    #models
    agent.actor.load_state_dict(torch.load(f"models/metasac_s{seed}_actor.h5"))
    agent.encoder.load_state_dict(torch.load(f"models/metasac_s{seed}_encoder.h5"))
    agent.critic.load_state_dict(torch.load(f"models/metasac_s{seed}_critic.h5"))

    os.makedirs("results", exist_ok=True)

    # OOD data

    winds = np.arange(0, 11, 1)  # 0 - 10 m/s
    delays = np.arange(0, 11, 1)  # 0 - 10 s

    # heatmap_data
    heatmap_data = np.zeros((len(winds), len(delays)))
    for i, w in enumerate(winds):
        for j, d in enumerate(delays):
            sr = evaluate_robustness(env, agent, wind=w, delay=d, eval_episodes=20)
            heatmap_data[i, j] = sr
            print(f"  Wind: {w} m/s | Delay: {d * 0.1:.1f} s -> Success Rate: {sr * 100:.1f}%")

    np.save(f"results/OOD_Heatmap_Data_s{seed}.npy", heatmap_data)

    boundary_delays = np.arange(0, 16, 1)
    boundary_sr = []

    for d in boundary_delays:
        sr = evaluate_robustness(env, agent, wind=4.0, delay=d, eval_episodes=30)
        boundary_sr.append(sr)
        print(f"  Fixed Wind: 4.0 m/s | Delay: {d * 0.1:.1f} s -> Success Rate: {sr * 100:.1f}%")

    np.save(f"results/Robustness_Boundary_Data_s{seed}.npy", np.array(boundary_sr))


if __name__ == "__main__":
    algos = ["MetaSAC"]

    for seed in SEEDS:

        train_metasac(seed)
        train_withoutlesson(seed)
        train_withoutper(seed)
        train_withoutgru(seed)
        train_withoutcouple(seed)

        for algo in algos:
            test_model_and_save_data(algo, seed)
        collect_ood_and_boundary_data(seed)

    print("\n🎉 全部数据生成完毕！所有 `.npy` 文件已存放至 `results/` 目录！")
