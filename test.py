import torch
from Feature_extractor import FeatureExtractor

B, K = 8, 512
dummy = {
    "images" : torch.rand(B, 2, 64, 64),
    "actions": torch.rand(B, K, 33)
}
net = FeatureExtractor()
logits, value = net(dummy)
print(logits.shape)   # torch.Size([8, 512])
print(value.shape)    # torch.Size([8, 1])
