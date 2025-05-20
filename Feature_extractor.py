import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    • CNN over the (2, 64, 64) images   → 64-d vector g
    • MLP over each (33,) action row    → 32-d vector aᵢ
    • [g ; aᵢ] → 1 logit
    • Value head uses g only
    """
    def __init__(self, observation_space, hidden=64):
        super().__init__()

        img_shape     = observation_space["images"].shape   # (2,64,64)
        action_dim    = observation_space["actions"].shape[-1]  # 33

        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, hidden), nn.ReLU(),
        )
        self.act_mlp   = nn.Sequential(
            nn.Linear(action_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
        )
        self.fc_logits = nn.Linear(hidden + 32, 1)
        self.fc_value  = nn.Linear(hidden, 1)

        # SB3 wants this even if we never use it afterwards
        self.features_dim = hidden        # arbitrary non-zero int

    def forward(self, obs):
        img = torch.as_tensor(obs["images"], dtype=torch.float32)  # (B,2,64,64)
        g   = self.cnn(img)                                        # (B,64)

        acts = torch.as_tensor(obs["actions"], dtype=torch.float32)  # (B,K,33)
        B, K, _ = acts.shape
        a_emb   = self.act_mlp(acts.view(B * K, -1)).view(B, K, -1) # (B*K, 32)

        g_exp   = g.unsqueeze(1).expand(-1, K, -1)
        fused   = torch.cat([g_exp, a_emb], dim=-1)                # (B,K,96)

        logits  = self.fc_logits(fused).squeeze(-1)                # (B,K)
        value   = self.fc_value(g)                                 # (B,1)
        return logits, value