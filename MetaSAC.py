import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.capacity, self.write, self.size_tracker = capacity, 0, 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done, gt_params):
        idx = self.write + self.capacity - 1
        self.data[self.write] = (state, action, reward, next_state, done, gt_params)
        change = self.max_priority - self.tree[idx]
        self.tree[idx] = self.max_priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
        self.write = (self.write + 1) % self.capacity
        if self.size_tracker < self.capacity: self.size_tracker += 1

    def sample(self, batch_size, beta=0.4):
        batch, idxs, priorities = [], [], []
        segment = self.tree[0] / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx = 0
            while 2 * idx + 1 < len(self.tree):
                left, right = 2 * idx + 1, 2 * idx + 2
                if s <= self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left];
                    idx = right
            priorities.append(self.tree[idx])
            batch.append(self.data[idx - self.capacity + 1])
            idxs.append(idx)
        is_weights = np.power(self.size_tracker * (np.array(priorities) / self.tree[0]), -beta)
        is_weights /= is_weights.max()
        s, a, r, ns, d, gt = zip(*batch)
        return np.array(s), np.array(a), np.array(r).reshape(-1, 1), np.array(ns), np.array(d).reshape(-1, 1), np.array(
            gt), idxs, np.array(is_weights).reshape(-1, 1)

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + 0.01) ** 0.6
            change = p - self.tree[idx]
            self.tree[idx] = p
            self.max_priority = max(self.max_priority, p)
            while idx != 0:
                idx = (idx - 1) // 2
                self.tree[idx] += change

    def size(self):
        return self.size_tracker

    def decay_priorities(self, decay_factor=0.3):
        """
        When switching courses, the priority of historical experiences in the pool is proportionally reduced.
        Prevent a large number of high error crash samples from instantly occupying the sampling queue in the new stage.
        """
        # Decay all leaf nodes (underlying nodes that store specific sample priorities)
        for i in range(self.capacity - 1, 2 * self.capacity - 1):
            if self.tree[i] > 0:
                self.tree[i] *= decay_factor

        # From bottom to top, recalculate the parent node interval and
        for i in range(self.capacity - 2, -1, -1):
            self.tree[i] = self.tree[2 * i + 1] + self.tree[2 * i + 2]

        # Gently reset the default maximum priority when inserting new samples
        self.max_priority = max(1.0, self.max_priority * decay_factor)
        print(f"♻️ [PER] 经验回放池优先级已执行温和衰减 (当前基准 Max Priority: {self.max_priority:.2f})")


# Dual decoupling encoder
class DualContextEncoder(nn.Module):
    def __init__(self, state_feat_dim=24, action_feat_dim=3, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.gru_state = nn.GRU(input_size=state_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.gru_action = nn.GRU(input_size=action_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.predictor = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 5))

    def forward(self, state_seq, action_seq):
        _, hn_s = self.gru_state(state_seq)
        _, hn_a = self.gru_action(action_seq)
        merged = torch.cat([hn_s.squeeze(0), hn_a.squeeze(0)], dim=1)
        z_raw = self.fc(merged)
        z_norm = self.layer_norm(z_raw)
        z = torch.tanh(z_norm)
        return z


class MetaActor(nn.Module):
    def __init__(self, state_dim=24, latent_dim=16, action_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + latent_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)

    def forward(self, state, z):
        x = F.relu(self.fc2(F.relu(self.fc1(torch.cat([state, z], 1)))))
        mu, std = self.fc_mu(x), F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        samp = dist.rsample()
        log_prob = dist.log_prob(samp).sum(-1, keepdim=True) - torch.log(1 - torch.tanh(samp).pow(2) + 1e-7).sum(-1,
                                                                                                                 keepdim=True)
        return torch.tanh(samp), log_prob


class MetaCritic(nn.Module):
    def __init__(self, state_dim=24, latent_dim=16, action_dim=3):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + latent_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256),
                                nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + latent_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256),
                                nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state, z, action):
        x = torch.cat([state, z, action], 1)
        return self.q1(x), self.q2(x)


class DualMetaSAC:
    def __init__(self, device):
        self.device = device
        self.state_dim, self.latent_dim = 24, 16
        self.state_H, self.action_H = 5, 10

        # Meta Network
        self.encoder = DualContextEncoder().to(device)
        self.actor = MetaActor().to(device)
        self.critic = MetaCritic().to(device)

        # Introduce target encoder
        self.target_encoder = DualContextEncoder().to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_critic = MetaCritic().to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.gamma, self.tau = 0.99, 0.005

    def _split_obs(self, meta_obs):
        s_hist_flat = meta_obs[:, :120]
        a_hist_flat = meta_obs[:, 120:150]
        s_hist = s_hist_flat.view(-1, 5, 24)
        a_hist = a_hist_flat.view(-1, 10, 3)
        obs = s_hist[:, -1, :]
        return obs, s_hist, a_hist

    def take_action(self, meta_obs, deterministic=False, is_level_0=False):
        meta_obs = torch.tensor(np.array([meta_obs]), dtype=torch.float).to(self.device)
        obs, s_hist, a_hist = self._split_obs(meta_obs)
        with torch.no_grad():
            z = self.encoder(s_hist, a_hist)
            if deterministic:
                x = F.relu(self.actor.fc2(F.relu(self.actor.fc1(torch.cat([obs, z], 1)))))
                action = torch.tanh(self.actor.fc_mu(x))
            else:
                action, _ = self.actor(obs, z)
        return action[0].cpu().numpy()

    def update(self, batch):
        s, a, r, ns, d, gt, idxs, weights = [
            torch.tensor(x, dtype=torch.float).to(self.device) if not isinstance(x, list) else x for x in batch]

        obs, s_hist, a_hist = self._split_obs(s)
        n_obs, n_s_hist, n_a_hist = self._split_obs(ns)

        # Update Encoder
        z = self.encoder(s_hist, a_hist)
        pred_env = self.encoder.predictor(z)
        # Retain the prediction error of each sample for PER
        pred_loss_unreduced = torch.mean((pred_env - gt) ** 2, dim=1)
        pred_loss = torch.mean(pred_loss_unreduced)

        z_l2_loss = torch.mean(z ** 2)
        total_encoder_loss = pred_loss + 0.05 * z_l2_loss

        self.encoder_opt.zero_grad()
        total_encoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.encoder_opt.step()

        z_det = z.detach()

        # Calculate target Q
        with torch.no_grad():
            # Calculate the latent variables for the next state using target_decoder
            n_z_raw = self.target_encoder(n_s_hist, n_a_hist)
            n_z_det = n_z_raw.detach()

            n_a, logp_next = self.actor(n_obs, n_z_det)
            q1_t, q2_t = self.target_critic(n_obs, n_z_det, n_a)
            target_q = r + (1 - d) * self.gamma * (torch.min(q1_t, q2_t) - self.log_alpha.exp() * logp_next)

        # Update Critic
        q1, q2 = self.critic(obs, z_det, a)
        td_error1 = target_q - q1
        td_error2 = target_q - q2
        td_error_min = torch.min(td_error1, td_error2).detach()

        asymmetric_errors = torch.where(td_error_min < 0,
                                        torch.abs(td_error_min) * 2.0,
                                        torch.abs(td_error_min))

        # Composite PER priority: control error+prediction error
        combined_priority = asymmetric_errors + 0.5 * pred_loss_unreduced.unsqueeze(1).detach()
        abs_errors = combined_priority.cpu().numpy().flatten()

        critic_loss = torch.mean(weights * (F.smooth_l1_loss(q1, target_q, reduction='none') +
                                            F.smooth_l1_loss(q2, target_q, reduction='none')))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # Update Actor & Alpha
        new_a, logp = self.actor(obs, z_det)
        q1_new, q2_new = self.critic(obs, z_det, new_a)

        actor_loss = torch.mean(self.log_alpha.exp().detach() * logp - torch.min(q1_new, q2_new))

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        alpha_loss = torch.mean(-self.log_alpha.exp() * (logp + 3.0).detach())
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            self.log_alpha.data.clamp_(min=-4.6, max=0.0)

        # Soft Update Target Network
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Soft Update target_encoder
        for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return abs_errors, critic_loss.item(), pred_loss.item()