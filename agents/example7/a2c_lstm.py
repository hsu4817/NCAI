import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from nle import nethack

#Referenced from NLE https://github.com/facebookresearch/nle/blob/main/nle/agent/agent.py
class Crop(nn.Module):
    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)[
                     None, :
                     ].expand(self.height_target, -1)
        height_grid = self._step_to_range(2 / (self.height - 1), height_target)[
                      :, None
                      ].expand(-1, self.width_target)

        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def _step_to_range(self, delta, num_steps):
        return delta * torch.arange(-num_steps // 2, num_steps // 2)

    def forward(self, inputs, coordinates):
        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
                .squeeze(1)
                .long()
        )

class A2C_LSTM(nn.Module):
    def __init__(self, crop_dim=15, final_layer_dims=256):
        super(A2C_LSTM, self).__init__()

        self.glyph_shape = (21, 79)
        self.num_actions = 23
        self.h = self.glyph_shape[0]
        self.w = self.glyph_shape[1]

        self.glyph_crop = Crop(self.h, self.w, crop_dim, crop_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.glyph_flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*11*11, out_features=final_layer_dims),
            nn.ReLU(),
        )

        self.actor = nn.Linear(in_features=final_layer_dims, out_features=self.num_actions)
        self.critic = nn.Linear(in_features=final_layer_dims, out_features=1)

        self.lstm = nn.LSTMCell(final_layer_dims, final_layer_dims)

    def forward(self, observed_glyphs, observed_stats, hx, cx):
        coordinates = observed_stats[:, :2]
        x_glyphs = self.glyph_crop(observed_glyphs, coordinates).unsqueeze(1).float()
        x_glyphs = self.cnn(x_glyphs)
        x_glyphs = self.glyph_flatten(x_glyphs)
        x = self.fc(x_glyphs)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        actor = Categorical(logits=self.actor(x))
        critic = self.critic(x)
        
        return actor, critic, hx, cx