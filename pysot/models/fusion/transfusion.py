from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch.nn.functional as F
import torch
from torch import nn


class transformerfusion(nn.Module):
    def __init__(self):
        super(transformerfusion, self).__init__()
        self.block0 = Block()

    def forward(self, down_x):
        x0 = down_x[0]
        x1 = down_x[1]
        x2 = down_x[2]
        x3 = down_x[3]
        x4 = down_x[4]
        nd_01 = self.block0(x0, x1)
        nd_12 = self.block0(x1, x2)
        nd_23 = self.block0(x2, x3)
        nd_34 = self.block0(x3, x4)
        f = nd_01 + nd_12 + nd_23 + nd_34
        return f


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        channels = 256
        self.out_channels = channels // 4
        self.a_key = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(), )
        self.a_query = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(), )
        self.a_value = nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                                 padding=0)
        self.a_W = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels * 2, kernel_size=1, stride=1,
                             padding=0)
        self.b_key = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(), )
        self.b_query = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(), )
        self.b_value = nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                                 padding=0)
        self.b_W = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels * 2, kernel_size=1, stride=1,
                             padding=0)
        self.gate_a = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(), )

    def forward(self, a, b):
        ns = self.fuse(a, b)
        return ns

    def fuse(self, a, b):
        a_m = a
        b_m = b
        adapt_channels = self.out_channels
        batch_size = a_m.size(0)
        a_query = self.a_query(a_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        a_key = self.a_key(a_m).view(batch_size, adapt_channels, -1)
        a_value = self.a_value(a_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        batch_size = b_m.size(0)
        b_query = self.b_query(b_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        b_key = self.b_key(b_m).view(batch_size, adapt_channels, -1)
        b_value = self.b_value(b_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        a_sim_map = torch.matmul(b_query, a_key)
        a_sim_map = F.softmax(a_sim_map, dim=-1)
        a_context = torch.matmul(a_sim_map, a_value)
        a_context = a_context.permute(0, 2, 1).contiguous()
        a_context = a_context.view(batch_size, self.out_channels, *a_m.size()[2:])
        a_context = self.a_W(a_context)
        b_sim_map = torch.matmul(a_query, b_key)
        b_sim_map = F.softmax(b_sim_map, dim=-1)
        b_context = torch.matmul(b_sim_map, b_value)
        b_context = b_context.permute(0, 2, 1).contiguous()
        b_context = b_context.view(batch_size, self.out_channels, *b_m.size()[2:])
        b_context = self.b_W(b_context)
        cat_fea = torch.cat([b_context, a_context], dim=1)
        attention_vector_a = self.gate_a(cat_fea)
        attention_vector_T = 1 - attention_vector_a
        ns = a * attention_vector_a + b * attention_vector_T
        return ns
