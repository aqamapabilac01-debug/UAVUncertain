import numpy as np
import gym
import collections
import gym
from gym import spaces

# Introduce map generation function
from UAVUncertain.Map0 import generate_shanghai_logistics_map

#Smooth posterior mesh (for training)
def generate_smooth_trajectory(waypoints_3d, resolution=0.5, smoothing=30.0):
    from scipy.interpolate import splprep, splev
    valid_idx = [0]
    for i in range(1, len(waypoints_3d)):
        if np.linalg.norm(waypoints_3d[i] - waypoints_3d[valid_idx[-1]]) > 0.5:
            valid_idx.append(i)
    pts = waypoints_3d[valid_idx]
    tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=smoothing, k=3)
    u_fine = np.linspace(0, 1, int(len(pts) * 10))
    x, y, z = splev(u_fine, tck)
    fine_pts = np.vstack((x, y, z)).T
    diffs = np.linalg.norm(np.diff(fine_pts, axis=0), axis=1)
    cum_dist = np.insert(np.cumsum(diffs), 0, 0)
    total_len = cum_dist[-1]
    s_even = np.arange(0, total_len, resolution)
    if s_even[-1] != total_len: s_even = np.append(s_even, total_len)
    even_pts = np.zeros((len(s_even), 3))
    for i in range(3): even_pts[:, i] = np.interp(s_even, cum_dist, fine_pts[:, i])
    tangents = np.zeros_like(even_pts)#计算切线
    tangents[:-1] = even_pts[1:] - even_pts[:-1]
    tangents[-1] = tangents[-2]
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-6)
    return even_pts, tangents, s_even

#Real posterior mesh (for testing)
def generate_linear_trajectory(waypoints_3d, resolution=0.5):
    even_pts = []
    for i in range(len(waypoints_3d) - 1):
        p1 = waypoints_3d[i]
        p2 = waypoints_3d[i + 1]
        dist = np.linalg.norm(p2 - p1)
        if dist == 0:
            continue

        num_pts = max(int(dist / resolution), 1)
        segment_pts = np.linspace(p1, p2, num_pts, endpoint=False)
        even_pts.extend(segment_pts)

    even_pts.append(waypoints_3d[-1])  # Endpoint
    even_pts = np.array(even_pts)

    tangents = np.zeros_like(even_pts)
    tangents[:-1] = even_pts[1:] - even_pts[:-1]
    tangents[-1] = tangents[-2]

    # Tangent vector normalization
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms != 0)

    # Calculate the path length s (used to calculate progress rewards)
    diffs = np.linalg.norm(np.diff(even_pts, axis=0), axis=1)
    s_even = np.insert(np.cumsum(diffs), 0, 0)

    return even_pts, tangents, s_even

import numpy as np


class SmoothDroneEnv(gym.Env):
    def __init__(self, raw_waypoints, use_smooth=True):
        super().__init__()
        if use_smooth:
            self.dense_path, self.tangents, self.path_s = generate_smooth_trajectory(np.array(raw_waypoints))
        else:
            self.dense_path, self.tangents, self.path_s = generate_linear_trajectory(np.array(raw_waypoints))

        self.dense_path_mirror = self.dense_path.copy()
        self.dense_path_mirror[:, 1] = -self.dense_path_mirror[:, 1]
        self.dm =self.dense_path_mirror
        self.tangents_mirror = self.tangents.copy()
        self.tangents_mirror[:, 1] = -self.tangents_mirror[:, 1]
        self.tm =self.tangents_mirror
        # [a_x, a_y, a_z]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.resolution, self.dt, self.g = 0.5, 0.1, 9.81
        self.base_m = 6.5;
        self.m = 6.5
        self.max_thrust, self.max_speed = 160.0, 10.0
        self.max_roll_pitch = np.deg2rad(45.0)
        self.max_yaw_rate = np.deg2rad(60.0)
        self.Kp_att = 5.0
        self.cruise_h = 50.0
        self.road_width = 25

        self.state = np.zeros(13)
        self.wind = np.zeros(3)
        self.prev_smoothed_action = np.zeros(3)
        self.prev_action = np.zeros(3)
        self.min_dist = 0
        self.proj_point = 0
        self.original_path = self.dense_path.copy()
        self.original_tangents = self.tangents.copy()
        self.tau_m = 0.03  # Response time constant of real motor (30ms)
        self.actual_tx_real = 0.0
        self.actual_ty_real = 0.0

    def reset(self, random_start=False, determine = False):

        if random_start == False or np.random.rand() > 0.5:
            self.dense_path = self.dense_path
            self.tangents = self.tangents
        else:
            self.dense_path = self.dense_path_mirror
            self.tangents = self.tangents_mirror
        if determine == True:
            self.dense_path = self.dm
            self.tangents = self.tm
        self.current_path = self.dense_path
        self.current_tangents = self.tangents

        self.start_pos = self.current_path[0].copy()
        self.target_pos = self.current_path[-1].copy()
        self.state = np.zeros(13)
        self.state[0:3] = self.start_pos
        self.state[3] = 1.0

        self.prev_s = 0
        self.max_s_achieved = 0
        self.steps_counter = 0
        self.current_closest_idx = 0
        self.min_dist = 0
        self.proj_point = 0

        self.actual_tx_real = 0.0
        self.actual_ty_real = 0.0

        return self._get_obs()

    def _get_closest_point(self, pos):
        """
        Local sliding window search for the nearest point to avoid progress jumps caused by path self intersection
        """
        search_radius = int(30.0 / self.resolution)
        start_idx = max(0, self.current_closest_idx - search_radius // 2)
        end_idx = min(len(self.dense_path), self.current_closest_idx + search_radius)

        window_pts = self.dense_path[start_idx:end_idx]
        dists = np.linalg.norm(window_pts - pos, axis=1)
        local_min_idx = np.argmin(dists)

        self.current_closest_idx = start_idx + local_min_idx
        min_dist = dists[local_min_idx]
        proj_point = self.dense_path[self.current_closest_idx]

        return min_dist, proj_point

    def step(self, action):

        self._physics_step(action)
        self.steps_counter += 1
        pos, vel, omega = self.state[0:3], self.state[7:10], self.state[10:13]
        self.min_dist, self.proj_point = self._get_closest_point(pos)

        # ==================== Reward ====================
        vel = self.state[7:10]
        current_tangent = self.current_tangents[self.current_closest_idx]

        # Effective advancement distance
        effective_progress = np.dot(vel, current_tangent) * self.dt

        # Potential energy reward (pushing distance along a smooth path)
        r_prog = effective_progress * 4.0
        curr_s = self.path_s[self.current_closest_idx]
        r_prog = (curr_s - self.prev_s) * 4.0  # 稍微拉大进度权重，对齐速度惩罚
        self.prev_s = curr_s

        # Enhanced path guidance
        r_guide = -0.1 * self.min_dist
        if self.min_dist > 15:
            r_guide = -1.5 - 0.1 * ((self.min_dist - 15) ** 2)

        # Speed incentive
        r_speed_push = 0
        dist_to_final = np.linalg.norm(pos - self.target_pos)
        speed = np.linalg.norm(vel)
        BRAKING_ZONE_RADIUS = 50
        if dist_to_final > BRAKING_ZONE_RADIUS and self.steps_counter > 30:
            target_v = 10.0
            if speed > 6.0:
                r_speed_push = -0.1 * (target_v - speed)
            else:
                r_speed_push = -0.02 * (target_v - speed) ** 2

        target_z = self.proj_point[2]
        r_time = -0.05
        r_ctrl = -0.05 * np.linalg.norm(omega)
        r_alt = -0.1 * abs(pos[2] - target_z)
        action_diff = np.linalg.norm(action - self.prev_action)
        self.prev_action = action
        r_smooth = -0.05 * action_diff

        step_reward = r_prog + r_time + r_ctrl + r_alt + r_guide + r_smooth + r_speed_push

        FINAL_ZONE_RADIUS = 5.0
        # Termination and constraint logic
        done = False
        info_status = "running"
        cost = 0.0

        is_crash = pos[2] < 40.0 or pos[2] > 60.0
        is_out_of_bounds = self.min_dist > self.road_width
        is_end_of_path = (self.current_closest_idx >= len(self.dense_path) - 2)

        if is_end_of_path:
            if dist_to_final < FINAL_ZONE_RADIUS:
                step_reward += 1000.0
                done = True
                info_status = "success"
            else:
                step_reward += 800.0
                cost = 1.0
                done = True
                info_status = "overshoot"

        elif is_crash or is_out_of_bounds:
            step_reward -= 300.0
            cost = 1.0
            done = True
            info_status = "crash" if is_crash else "out_of_bounds"

        if self.steps_counter >= 10000:
            done = True
            step_reward -= 300.0
            info_status = "timeout"

        # Status export

        env_info = {
            "cost": cost,
            "info_status": info_status,
            "is_success": (info_status == "success"),
            "physics_state": self.state.copy(),
            "logical_state": np.array([self.current_closest_idx, curr_s], dtype=np.float32)
        }

        return self._get_obs(), step_reward, done, env_info

    def _get_obs(self):
        pos, q, vel, omega = self.state[0:3], self.state[3:7], self.state[7:10], self.state[10:13]
        w, x, y, z = q

        # Rotation Matrix and Rotating Machine System Functions
        R = np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]])
        R_inv = R.T

        def to_body(vec): return R_inv @ vec

        # --- Basic dimensions (9 dimensions) ---
        body_vel = np.clip(to_body(vel) / self.max_speed, -1.2, 1.2)
        cte_vec = self.proj_point - pos
        body_cte_vec = np.clip(to_body(cte_vec) / 10.0, -1.0, 1.0)
        b_curr_tan = to_body(self.current_tangents[self.current_closest_idx])

        # --- Forward looking gaze position and tangent (6 dimensions) ---
        idx_25 = min(self.current_closest_idx + int(25.0 / self.resolution), len(self.current_path) - 1)
        b_tan_25 = to_body(self.current_tangents[idx_25])

        lookahead_pos = self.current_path[idx_25]
        lookahead_vec = lookahead_pos - pos
        body_lookahead_vec = np.clip(to_body(lookahead_vec) / 25.0, -1.2, 1.2)  # 真正的 Pure Pursuit 诱导向量

        # --- Physical organism self reflection perception (6 dimensions) ---
        body_omega = np.clip(omega / self.max_yaw_rate, -1.2, 1.2)  # 感受风扰的最佳途径
        world_gravity = np.array([0.0, 0.0, -1.0])
        body_gravity = to_body(world_gravity)  # 让网络知道绝对水平面在哪

        # --- Endpoint vector (3 dimensions) ---
        rel_target = self.target_pos - pos
        body_rel_target = np.clip(to_body(rel_target / 4000.0), -1.2, 1.2)

        # The total dimension of the concatenated observations
        obs = np.concatenate([
            body_vel, body_cte_vec, b_curr_tan,
            b_tan_25, body_lookahead_vec,
            body_omega, body_gravity,
            body_rel_target
        ]).astype(np.float32).flatten()

        return obs

    def _physics_step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # RL output physical meaning: forward acceleration of the body (X),
        # lateral acceleration of the body (Y), global vertical acceleration (Z)
        a_x_body, a_y_body, a_z = action[0] * 5.0, action[1] * 5.0, action[2] * 3.0
        num_substeps = 5
        dt_sub = self.dt / num_substeps

        for _ in range(num_substeps):
            w, x, y, z = self.state[3:7]
            curr_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))

            # Rotate the expected horizontal thrust of the aircraft system to the world coordinate system
            # (provided to the underlying engine for calculating the actual force)
            a_x_world = a_x_body * np.cos(curr_yaw) - a_y_body * np.sin(curr_yaw)
            a_y_world = a_x_body * np.sin(curr_yaw) + a_y_body * np.cos(curr_yaw)

            # Yaw control logic
            # Always align the nose smoothly with the global tangent direction of the current road,
            # decoupling translation and rotation
            tan = self.current_tangents[self.current_closest_idx]
            target_yaw = np.arctan2(tan[1], tan[0])

            # Calculate the yaw angle error and normalize it to [pi, pi]
            yaw_error = (target_yaw - curr_yaw + np.pi) % (2 * np.pi) - np.pi

            # Using P controller to track heading
            yaw_rate_cmd = np.clip(2.0 * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)

            # Thrust and attitude calculation using acceleration in the world coordinate system
            T_x, T_y, T_z = self.m * a_x_world, self.m * a_y_world, self.m * (a_z + self.g)
            throttle = np.clip(np.sqrt(T_x ** 2 + T_y ** 2 + T_z ** 2), 0, self.max_thrust)

            roll_cmd = np.clip(
                np.arcsin(np.clip((T_x * np.sin(curr_yaw) - T_y * np.cos(curr_yaw)) / (throttle + 1e-6), -1.0, 1.0)),
                -self.max_roll_pitch, self.max_roll_pitch)

            pitch_cmd = np.clip(np.arctan2(T_x * np.cos(curr_yaw) + T_y * np.sin(curr_yaw), T_z),
                                -self.max_roll_pitch, self.max_roll_pitch)

            curr_roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            curr_pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))

            # PD Controller
            real_Kp = 65.0
            real_Kd = 18.0

            tx = real_Kp * (roll_cmd - curr_roll) - real_Kd * self.state[10]
            ty = real_Kp * (pitch_cmd - curr_pitch) - real_Kd * self.state[11]
            tz = 12.0 * (yaw_rate_cmd - self.state[12])

            # Extract the current physical state
            pos, q, vel, omega = self.state[0:3], self.state[3:7], self.state[7:10], self.state[10:13]
            R = np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]])

            # Wind
            v_rel = vel - self.wind

            # f_Wind
            f_drag_world = -0.25 * v_rel * np.linalg.norm(v_rel)

            # Total F
            f_tot = np.array([0, 0, -9.81 * self.m]) + R @ np.array([0, 0, throttle]) + f_drag_world

            # Pendulum and Wind Sail Dynamics
            I_base = 0.08

            # Center of mass displacement
            dz = getattr(self, 'dz_com', 0.0)

            # I_new = I_base + m * d^2
            I_new = I_base + self.m * (dz ** 2)

            # Inertia ratio: represents the degree to which the flight control attitude response is "weakened"
            inertia_ratio = I_base / I_new

            # Wind Torque
            # body f_wind
            f_drag_body = R.T @ f_drag_world

            # tau = r × F
            tau_wind_x = -dz * f_drag_body[1]  # Roll
            tau_wind_y = dz * f_drag_body[0]  # Pitch
            # Pneumatic coupling
            v_rel_body = R.T @ v_rel
            c_flap = 0.003
            tau_flap_x = -c_flap * v_rel_body[1]
            tau_flap_y = c_flap * v_rel_body[0]
            # =====================================================================

            # Actual angular acceleration
            tx_real = tx * inertia_ratio + ((tau_wind_x + tau_flap_x) / I_new)
            ty_real = ty * inertia_ratio + ((tau_wind_y + tau_flap_y) / I_new)
            tz_real = tz * inertia_ratio

            # y(t) = y(t-1) + (dt / tau) * (u(t) - y(t-1))
            alpha_filter = dt_sub / (self.tau_m + dt_sub)

            self.actual_tx_real = self.actual_tx_real + alpha_filter * (tx_real - self.actual_tx_real)
            self.actual_ty_real = self.actual_ty_real + alpha_filter * (ty_real - self.actual_ty_real)

            # velocity integration
            vel_n = vel + f_tot / self.m * dt_sub
            if np.linalg.norm(vel_n) > self.max_speed:
                vel_n = (vel_n / np.linalg.norm(vel_n)) * self.max_speed

            # Position Points
            pos_n = pos + vel_n * dt_sub
            omega_n = omega + np.array([self.actual_tx_real, self.actual_ty_real, tz_real]) * dt_sub

            # Quaternion Integral
            q_dot = 0.5 * np.array([-x * omega[0] - y * omega[1] - z * omega[2], w * omega[0] + y * omega[2] - z * omega[1],
                                    w * omega[1] - x * omega[2] + z * omega[0], w * omega[2] + x * omega[1] - y * omega[0]])
            q_n = (q + q_dot * dt_sub) / np.linalg.norm(q + q_dot * dt_sub)

            # Update status
            self.state = np.concatenate([pos_n, q_n, vel_n, omega_n])

#Env Wrapper
class DualMetaContextWrapper(gym.Wrapper):
    def __init__(self, env, state_H=5, action_H=10):
        super().__init__(env)
        self.state_H = state_H
        self.action_H = action_H
        self.obs_dim = 24
        self.act_dim = 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(150,), dtype=np.float32)
        self.state_queue = collections.deque(maxlen=state_H)
        self.action_queue = collections.deque(maxlen=action_H)
        self.delay_buffer = collections.deque(maxlen=10)

        self.wind_range = 0.0
        self.mass_var = 0.0
        self.max_delay = 0

        # OU Wind
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.target_wind = np.zeros(3)

    def _sample_wind_vector(self):
        #Target wind speed vector
        if self.wind_range <= 1e-5:
            return np.zeros(3)
        w = np.random.uniform(-self.wind_range, self.wind_range, size=3)
        w[2] *= 0.3
        return w

    def reset(self, random_start=False, determine=False):
        # Quality disturbance
        m_p = self.unwrapped.base_m * np.random.uniform(0.0, self.mass_var)
        self.unwrapped.m = self.unwrapped.base_m + m_p
        # Changes in the center of mass of unmanned aerial vehicles caused by quality mounting
        self.unwrapped.dz_com = 0.05 * m_p

        # OU Non Gaussian Wind Field Initialization
        self.target_wind = self._sample_wind_vector()
        self.unwrapped.wind = self.target_wind.copy()

        # Delay Initialization
        self.curr_delay = np.random.randint(0, self.max_delay + 1) if self.max_delay > 0 else 0
        self.delay_buffer.clear()
        for _ in range(self.curr_delay):
            self.delay_buffer.append(np.zeros(self.act_dim))

        # 4. Env Reset
        obs = self.env.reset(random_start=random_start, determine=determine)

        # 5. Queue Initialization
        self.state_queue.clear()
        self.action_queue.clear()
        for _ in range(self.state_H):
            self.state_queue.append(obs.copy())
        for _ in range(self.action_H):
            self.action_queue.append(np.zeros(self.act_dim))
        self.unwrapped.actual_tx_real = 0.0
        self.unwrapped.actual_ty_real = 0.0
        return self._get_meta_obs(obs)

    def step(self, action):
        # Dynamic evolution of environmental parameters
        if self.wind_range > 1e-5 and np.random.rand() < 0.05:
            self.target_wind = self._sample_wind_vector()

        if self.max_delay > 0 and np.random.rand() < 0.01:
            new_delay = np.random.randint(0, self.max_delay + 1)
            if new_delay > self.curr_delay:
                for _ in range(new_delay - self.curr_delay):
                    self.delay_buffer.appendleft(action)
            elif new_delay < self.curr_delay and len(self.delay_buffer)>=2:
                for _ in range(self.curr_delay - new_delay):
                    self.delay_buffer.popleft()
            self.curr_delay = new_delay


        # OU
        if self.wind_range > 1e-5:
            wind_diff = self.target_wind - self.unwrapped.wind
            noise = np.random.randn(3) * self.ou_sigma * self.wind_range
            noise[2] *= 0.3
            self.unwrapped.wind += self.ou_theta * wind_diff + noise
            self.unwrapped.wind = np.clip(
                self.unwrapped.wind, [-self.wind_range, -self.wind_range, -self.wind_range * 0.3],
                [self.wind_range, self.wind_range, self.wind_range * 0.3]
            )

        # Action Delay
        self.action_queue.append(action)

        if self.curr_delay > 0:
            self.delay_buffer.append(action)
            exec_action = self.delay_buffer.popleft()
        else:
            exec_action = action

        # Interacting with the underlying environment
        obs, reward, done, info = self.env.step(exec_action)
        self.state_queue.append(obs.copy())

        # Ground Truth
        # Map the dimensions of the predicted target uniformly to the vicinity of [-1,1]
        norm_wind_x = self.unwrapped.wind[0] / 6.0
        norm_wind_y = self.unwrapped.wind[1] / 6.0
        norm_wind_z = self.unwrapped.wind[2] / 1.8  
        norm_mass = (self.unwrapped.m - self.unwrapped.base_m) / (self.unwrapped.base_m * 0.3 + 1e-6)
        norm_delay = self.curr_delay / 5.0

        gt_params = np.array([norm_wind_x, norm_wind_y, norm_wind_z, norm_mass, norm_delay], dtype=np.float32)
        info['gt_params'] = gt_params

        return self._get_meta_obs(obs), reward, done, info

    def _get_meta_obs(self, obs):
        state_arr = np.array(self.state_queue).flatten()
        action_arr = np.array(self.action_queue).flatten()
        return np.concatenate([state_arr, action_arr]).astype(np.float32)