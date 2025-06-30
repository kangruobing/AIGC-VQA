import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50


class Slowfast(nn.Module):
    def __init__(self):
        super().__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        """
        Args:
            frame_list: [[B, C, N, H, W], [B, C, N, H, W]]
        Returns:
            features: slow_feature: [B, 2048],
                      fast_feature: [B, 256]
        """
        x = self.feature_extraction(x)

        slow_feature = self.slow_avg_pool(x[0])
        fast_feature = self.fast_avg_pool(x[1])

        slow_feature = self.adp_avg_pool(slow_feature).flatten(1)
        fast_feature = self.adp_avg_pool(fast_feature).flatten(1)

        return slow_feature, fast_feature
    
"""
def test():
    model = Slowfast()

    x = torch.randn(2, 3, 32, 224, 224)
    frame_list = pack_pathway_output(x)

    slow_feature, fast_feature = model(frame_list)

    print(f"slow feature shape: {slow_feature.shape}")
    print(f"fast feature shape: {fast_feature.shape}")

if __name__ == "__main__":
    test()
"""