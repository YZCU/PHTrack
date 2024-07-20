from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch.nn.functional as F
import torch.nn as nn
import torch


def scl(rc, order):
    br = []
    b = rc.size()[0]
    ofr = [[None for j in range(b)] for i in range(5)]
    obi = order[0]
    ob = order[1]
    for i in range(5):
        select_three_band = ob[0, i * 3:i * 3 + 3]
        gg = rc[None, 0, select_three_band, :, :]
        ofr[i][0] = obi[0, i * 3:i * 3 + 3].detach().mean().item()
        for k in range(1, b):
            stbo = ob[k, i * 3:i * 3 + 3]
            gg = torch.cat((gg, rc[None, k, stbo, :, :]), dim=0)
            ofr[i][k] = obi[k, i * 3:i * 3 + 3].detach().mean().item()
        br.append(gg)
    return br, ofr


class cbr_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(cbr_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class db_3(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, padding, kernel_size=3, stride=1, groups=1, bias=False):
        super(db_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class msp(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(msp, self).__init__()
        self.dc_1 = db_3(mid_channels, mid_channels, dilation=1, padding=1)
        self.dc_2 = db_3(mid_channels, mid_channels, dilation=6, padding=6)
        self.dc_3 = db_3(mid_channels, mid_channels, dilation=12, padding=12)
        self.c_0 = cbr_1(in_channels, mid_channels)
        self.c4_0 = cbr_1(mid_channels, mid_channels)
        self.c5_0 = cbr_1(mid_channels, mid_channels)
        self.c_4 = cbr_1(mid_channels, mid_channels)
        self.c_5 = cbr_1(mid_channels, mid_channels)
        self.fd = cbr_1(4 * mid_channels, in_channels)

    def forward(self, x):
        x_1 = self.dc_1(self.c_0(x))
        x_2 = self.dc_2(self.c_0(x))
        x_3 = self.dc_3(self.c_0(x))
        x_0 = self.c_0(x)
        x_1_cos = self.c5_0(x_1 * torch.cos(x_1))
        x_1_sin = self.c4_0(x_1 * torch.sin(x_1))
        m_1 = self.c_4(x_1_sin * x_1_cos)
        s_1 = self.c_5(x_1_sin + x_1_cos)
        fus_1 = s_1 + m_1
        x_2_c = self.c5_0(x_2 * torch.cos(x_2))
        x_2_s = self.c4_0(x_2 * torch.sin(x_2))
        mn_2 = self.c_4(x_2_s * x_2_c)
        sn_2 = self.c_5(x_2_s + x_2_c)
        fus_2 = sn_2 + mn_2
        x_3_c = self.c5_0(x_3 * torch.cos(x_3))
        x_3_s = self.c4_0(x_3 * torch.sin(x_3))
        mn_3 = self.c_4(x_3_s * x_3_c)
        sn_3 = self.c_5(x_3_s + x_3_c)
        fus_3 = sn_3 + mn_3
        oc = torch.cat((fus_1, fus_2, fus_3, x_0), dim=1)
        out = self.fd(oc)
        return out


class MSC(nn.Module):
    def __init__(self):
        super(MSC, self).__init__()
        in_channels = 16
        mid_channels = 8
        self.msp = msp(in_channels=in_channels, mid_channels=mid_channels)
        self.mlp = nn.Sequential(nn.Linear(in_channels, in_channels, bias=True), nn.Tanh(),
                                 nn.Linear(in_channels, in_channels, bias=True), nn.Tanh())

    def forward(self, x):
        xx = x.mul(1 / 255.0).clamp(0.0, 1.0)
        x_assp = self.msp(xx)
        b, c, w, h = x_assp.size()
        x2 = x_assp.view(b, c, -1)
        x3 = x2.permute(0, 2, 1)
        ms_mlp = self.mlp(x3)
        ms_mlp_ = ms_mlp.transpose(1, 2)
        cm = torch.matmul(ms_mlp_, ms_mlp)
        for i in range(16):
            cm[:, i, i] = 0.0
        cm = F.normalize(cm, p=2, dim=-1)
        for i in range(16):
            cm[:, i, i] = 0.0
        w = torch.norm(cm, p=1, dim=-1)
        w_reshape = w.contiguous().view(w.shape[0], w.shape[1], 1, 1)
        x_att = w_reshape.expand_as(x)
        x_cg = xx * x_att
        x_cg = x_cg.view(x_cg.shape[0], x_cg.shape[1], -1)
        x_orig = xx.view(xx.shape[0], xx.shape[1], -1)
        cg_list = [x_orig, x_cg]
        order_Y = torch.sort(w, dim=-1, descending=True, out=None)
        fc_list, w_list = scl(x, order_Y)
        return fc_list, w_list, order_Y, cg_list
