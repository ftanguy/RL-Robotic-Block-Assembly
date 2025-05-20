"""
Wrapps the AssemblyEnv in a gym interface so that it can be used with stable_baselines3
"""


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import logging

import matplotlib.pyplot as plt

# ---- project imports -------------------------------------------------------
from assembly_env import AssemblyEnv
from tasks import Bridge, Tower, DoubleBridge, TripleBridge
from utils.logger_utils import get_logger

MAX_ACTIONS = 512 # Upper limit on the number of possible actions
ACTION_ENCODING = 33 # Size of each action encoding

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from Feature_extractor import FeatureExtractor

class BAActorCritic(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         **kwargs,
                         features_extractor_class = FeatureExtractor,
                         features_extractor_kwargs = {},
                         net_arch=[])      # we supply our own heads

        self.action_net = nn.Identity()
        self.value_net  = nn.Identity()

        # Custom forward that simply delegates to the extractor
    def forward(self, obs, deterministic: bool = False, action_masks=None):
        """
        :param obs: tensor dict from MaskablePPO
        :param deterministic: bool
        :param action_masks: np.ndarray | None  - shape (B, K)
        """
        # 1) FeatureExtractor gives us raw logits for *every* action row
        logits, values = self.extract_features(obs)         # (B,K) , (B,1)

        # 2) Build a categorical distribution
        dist = self.action_dist.proba_distribution(action_logits=logits)

        # 3) Apply masking (sets prob=0 for illegal rows)
        if action_masks is not None:
            dist.apply_masking(action_masks)

        # 4) Sample or take arg-max
        actions = dist.get_actions(deterministic=deterministic)   # (B,)

        # 5) Log-probability of chosen actions
        log_prob = dist.log_prob(actions)

        return actions, values, log_prob
    
    def predict_values(self, obs):
        """
        Return V(s) given observations `obs`.
        Called by MaskablePPO when computing returns.
        """
        _, values = self.features_extractor(obs)     # tuple -> take value head
        return values                                # tensor (B,1)

    def evaluate_actions(self, obs, actions, action_masks=None):
        """
        Return (value, log_prob, entropy) for the chosen `actions`.
        """
        logits, values = self.features_extractor(obs)        # (B,K) , (B,1)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)

        log_prob = dist.log_prob(actions)                    # (B,)
        entropy  = dist.entropy()                            # (B,)
        return values, log_prob, entropy

    def get_distribution(self, obs, action_masks=None):
        """
        Build a `MaskedCategorical` from our per-candidate logits.
        Called by predict(), _predict() and elsewhere.
        """
        logits, _ = self.features_extractor(obs)        # (B, K)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    def _predict(self, obs, deterministic: bool = False, action_masks=None):
        """
        Low-level helper used by predict() and evaluation code.
        Must return *only* the chosen actions tensor.
        """
        dist = self.get_distribution(obs, action_masks)
        return dist.get_actions(deterministic=deterministic)

class BlockAssemblyGym(gym.Env):

    def __init__(self, task, num_block_offsets: int = 1, render = False, level = logging.INFO):
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.setLevel(level)

        self.task_name = task
        self.task = self.make_task(self.task_name, 2)
        self.backend = AssemblyEnv(self.task,level=level)

        self.num_block_offsets = num_block_offsets
        self.max_actions = MAX_ACTIONS
        self.render_enabled = render
        self.fig, self.ax = None, None

        self._actions_encoding = np.zeros((MAX_ACTIONS,ACTION_ENCODING), dtype=np.float32)
        self._action_list = []
        self._mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.max_actions)

        h, w = self.backend.img_size
        
        self.observation_space = gym.spaces.Dict({
            "images" : gym.spaces.Box(0, 1, shape=(3, h, w), dtype=np.float32),
            "actions" : gym.spaces.Box(-2, 2, shape=(MAX_ACTIONS,ACTION_ENCODING), dtype=np.float32)
        })
  
    def _refresh_actions(self):
        acts = self.backend.available_actions(num_block_offsets=self.num_block_offsets)
        self._action_list = acts

        self._actions_encoding = np.zeros((MAX_ACTIONS,ACTION_ENCODING), dtype=np.float32)
        self._mask = np.zeros(MAX_ACTIONS, dtype=np.float32)

        for i, action in enumerate(acts):
            vec = self.backend.encode_action(action)

            self._actions_encoding[i] = vec
            self._mask[i] = 1


    def action_masks(self):
        return self._mask

    def _get_obs(self):
        state_img  = self.backend.state_feature          # (1,64,64)    
        reward_img = self.backend.reward_feature.unsqueeze(0)  # (1,64,64)
        obstacle_img = self.backend.obstacle_feature
        images = torch.cat([state_img, reward_img, obstacle_img], dim=0).float()  # (2,64,64)
        self._refresh_actions()
        
        return {"images" : images,
                "actions" : self._actions_encoding
                }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.task = self.make_task(self.task_name,np.random.randint(2,4))
        self.backend = AssemblyEnv(self.task, level=self.logger.level)
        self.backend.reset()
        obs = self._get_obs()
        return obs, {}

    def step(self, index : int):
        """
        Returns: obs, reward, terminted, truncated, info
        """
        action = self._action_list[index]
        obs, reward, terminated = self.backend.step(action)
        self.logger.debug(f"Reward: {reward}")
        obs = self._get_obs()

        truncated = False  

        if self.render_enabled:
            self.render()
        return obs, reward, terminated, truncated, {}

    def render(self):
        from rendering import plot_assembly_env

        if self.fig is None:
            self.fig, axes = plt.subplot_mosaic(
                """
                AB
                CD
                """,
                figsize=(10, 8),
            )
            self.ax_geom      = axes["A"]
            self.ax_state     = axes["B"]
            self.ax_reward    = axes["C"]
            self.ax_obstacles = axes["D"]

            self.ax_geom.set_title("Assembly")
            self.ax_state.set_title("State")
            self.ax_reward.set_title("Reward")
            self.ax_obstacles.set_title("Obstacles")

            # remove ticks on the heatmaps
            for ax in (self.ax_state, self.ax_reward, self.ax_obstacles):
                ax.set_xticks([])
                ax.set_yticks([])

            # --- create the image artists with correct shapes ----------
            # state_feature is (1,H,W) → squeeze to (H,W)
            img_state = self.backend.state_feature.squeeze(0).numpy()
            self.img_state = self.ax_state.imshow(
                img_state,
                cmap="gray",
                interpolation="none",
                vmin=0.0, vmax=1.0,
            )

            # reward_feature is (H,W)
            self.img_reward = self.ax_reward.imshow(
                self.backend.reward_feature.numpy(),
                cmap="viridis",
                interpolation="none",
            )

            # obstacle_feature is (1,H,W) → squeeze to (H,W)
            img_obs = self.backend.obstacle_feature.squeeze(0).numpy()
            self.img_obstacles = self.ax_obstacles.imshow(
                img_obs,
                cmap="Reds",
                interpolation="none",
                vmin=0.0, vmax=1.0,
            )

        # -------- update each frame ---------------------------------------
        self.ax_geom.clear()
        plot_assembly_env(self.backend, fig=self.fig, ax=self.ax_geom, task=self.task)

        # update the heatmaps:
        self.img_state.set_data(self.backend.state_feature.squeeze(0).numpy())
        self.img_reward.set_data(self.backend.reward_feature.numpy())
        self.img_obstacles.set_data(self.backend.obstacle_feature.squeeze(0).numpy())

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


    def close(self):
        pass

    def make_task(self, name: str, num_stories: int):
        if name == "bridge":
            return Bridge(num_stories=num_stories)
        if name == "tower":
            # tower has a list of (x, height) targets – we centre at 0
            targets = [(0, h) for h in range(1, num_stories + 1)]
            return Tower(targets)
        if name == "double_bridge":
            return DoubleBridge(num_stories=num_stories)
        
        if name == "triple_bridge":
            return TripleBridge(num_stories=num_stories)
        raise ValueError(f"Unknown task '{name}'.")
    
    def print_available_actions(self):
        """Prints all available actions one per line with index."""
        actions = self.backend.available_actions()
        for i, a in enumerate(actions):
            print(f"[{i:03}] target_block={a.target_block}, "
                f"target_face={a.target_face}, shape={a.shape}, "
                f"face={a.face}, offset_x={a.offset_x}")
