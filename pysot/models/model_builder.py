from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.loss_car import make_phtrack_loss_evaluator
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from pysot.models.channelsplit import get_BS
from pysot.models.backbone import get_backbone
from pysot.models.neck import get_neck
from pysot.models.fusion import get_FS
from pysot.models.head.car_head import CARHead


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.zf = None

        self.channelsplit = get_BS(cfg.BS.TYPE)

        self.backbone = get_backbone(cfg.BACKBONE.TYPE, **cfg.BACKBONE.KWARGS)

        self.neck = get_neck(cfg.ADJUST.TYPE, **cfg.ADJUST.KWARGS)

        self.xcorr_depthwise = xcorr_depthwise

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

        self.fs = get_FS(cfg.FS.TYPE)

        self.car_head = CARHead(cfg, 256)

        self.loss_evaluator = make_phtrack_loss_evaluator(cfg)

    def template(self, z):

        bs_z = self.channelsplit(z)

        for i in range(len(bs_z[0])):
            z = bs_z[0][i]

            zf = self.backbone(z)

            zf = self.neck(zf)

            if i == 0:
                zff = [zf]
            else:
                zff.append(zf)

        self.zf = zff

    def track(self, x):

        bs_x = self.channelsplit(x)

        for i in range(len(bs_x[0])):

            x = bs_x[0][i]

            xf = self.backbone(x)

            xf = self.neck(xf)

            features = self.xcorr_depthwise(xf[0], self.zf[i][0])

            for j in range(len(xf) - 1):
                features_new = self.xcorr_depthwise(xf[j + 1], self.zf[i][j + 1])

                features = torch.cat([features, features_new], 1)

            f_down = self.down(features)

            if i == 0:
                f_list = [f_down]
            else:
                f_list.append(f_down)

        f_f = self.fs(f_list)

        cls, loc, cen = self.car_head(f_f)
        bs_order = bs_x[2][1][0][0:3]

        w_16band = bs_x[2][0]

        w_std = torch.std(w_16band)

        return {'cls': cls, 'loc': loc, 'cen': cen, 'bs_order': bs_order, 'w_std': w_std}

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """
        only used in training
        输出为损失
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        template_cs = self.channelsplit(template)
        search_cs = self.channelsplit(search)

        for i in range(5):
            template_3 = template_cs[0][i]
            search_3 = search_cs[0][i]

            zf0 = self.backbone(template_3)

            xf0 = self.backbone(search_3)

            zf = self.neck(zf0)

            xf = self.neck(xf0)

            f_0 = self.xcorr_depthwise(xf[0], zf[0])
            f_1 = self.xcorr_depthwise(xf[1], zf[1])
            f_2 = self.xcorr_depthwise(xf[2], zf[2])

            fea = torch.cat([f_0, f_1, f_2], dim=1)

            f_down = self.down(fea)

            if i == 0:
                f_five = [f_down]
            else:
                f_five.append(f_down)

        f_fs = self.fs(f_five)

        cls, loc, cen = self.car_head(f_fs)

        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)

        cls_loss, loc_loss, cen_loss = self.loss_evaluator(locations, cls, loc, cen, label_cls, label_loc)

        t_cg_loss = F.l1_loss(template_cs[3][0], template_cs[3][1])

        s_cg_loss = F.l1_loss(search_cs[3][0], search_cs[3][1])
        cg_loss = (t_cg_loss.data + s_cg_loss.data) / 2

        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss + cfg.TRAIN.CG_WEIGHT * cg_loss

        outputs = {'total_loss': total_loss, 'cls_loss': cls_loss, 'loc_loss': loc_loss, 'cen_loss': cen_loss,
                   'cg_loss': cg_loss}

        return outputs
