import torch
import torch.nn as nn
import torch.nn.functional as F


class L2CAILayer(nn.Module):
    def __init__(self, channels):
        super(L2CAILayer, self).__init__()
        self.ic, self.pc = channels
        rc = self.ic // 4
        self.conv1 = nn.Sequential(nn.Conv2d(self.pc, self.ic, 1),
                                   nn.BatchNorm2d(self.ic),
                                   nn.ReLU(True))
        # self.fc1 = nn.Linear(self.ic, rc)
        self.fc1 = nn.Conv2d(self.ic, rc, kernel_size=1)
        self.fc2 = nn.Conv2d(self.pc, rc, kernel_size=1)
        self.fc3 = nn.Conv2d(rc, 1, kernel_size=1)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feas: <Tensor, N, C> image_feature conv+bn
            point_feas: <Tensor, N, C'> point_feature conv+bn+relu

        Returns:

        """
        img_feats = img_feats.contiguous()
        point_feats = point_feats.contiguous()
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        # att = att.unsqueeze(1).view(1, 1, -1)  # B1N

        # img_feats_c = img_feats.unsqueeze(0).transpose(1, 2).contiguous()
        # 依据图像特征和点云特征的相关程度筛选图像特征
        out = self.conv1(point_feats) * att

        return out


class L2CFusion(nn.Module):

    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(L2CFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes_I + inplanes_I, out_channels=outplanes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.l2c_ai_layer = L2CAILayer(channels=[inplanes_I, inplanes_P])

    def forward(self, point_features, img_features):
        """
        point_feature: 点云特征 [B, C, N] conv+bn+relu
        img_feature: 图像特征 [B, N, C]  conv+bn
        """

        l2c_features = self.l2c_ai_layer(img_features, point_features)  # [B, C, N]
        fusion_features = torch.cat([img_features, l2c_features], dim=1)
        fusion_features = self.bn1(self.conv1(fusion_features))

        return F.relu(fusion_features)



class IA_Layer(nn.Module):
    def __init__(self, channels, return_att=False):
        """
        ic: [64, 128, 256, 512]
        pc: [96, 256, 512, 1024]
        """
        super(IA_Layer, self).__init__()

        self.return_att = return_att
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU(True))
        # self.fc1 = nn.Linear(self.ic, rc)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.ic),
            nn.ReLU(True),
            nn.Linear(self.ic, rc)
        )
        self.fc2 = nn.Linear(self.pc, rc)
        # self.fc2 = nn.Sequential(
        #     nn.BatchNorm1d(self.pc),
        #     nn.ReLU(True),
        #     nn.Linear(self.pc, rc)
        # )
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feas: <Tensor, N, C> image_feature conv+bn
            point_feas: <Tensor, N, C'> point_feature conv+bn+relu

        Returns:

        """
        img_feats = img_feats.contiguous()
        point_feats = point_feats.contiguous()
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        att = att.unsqueeze(1).view(1, 1, -1)  # B1N

        img_feats_c = img_feats.unsqueeze(0).transpose(1, 2).contiguous()
        img_feas_new = self.conv1(img_feats_c)
        # 依据图像特征和点云特征的相关程度筛选图像特征
        out = img_feas_new * att

        return out


class C2LFusion(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes, return_att=False):
        """
        inplanes_I: [64, 128, 256, 512]
        inplanes_P: [96, 256, 512, 1024]
        outplanes: [96, 256, 512, 1024]
        """
        super(C2LFusion, self).__init__()

        self.return_att = return_att

        self.ai_layer = IA_Layer(channels=[inplanes_I, inplanes_P], return_att=return_att)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        """
        point_feature: 点云特征 [B, C, N] conv+bn+relu
        img_feature: 图像特征 [B, N, C]  conv+bn
        """

        img_features = self.ai_layer(img_features, point_features)  # [B, C, N]
        # print("img_features:", img_features.shape)
        point_feats = point_features.unsqueeze(0).transpose(1, 2)
        # 将筛选的图像特征与点云特征直接拼接
        fusion_features = torch.cat([point_feats, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        fusion_features = fusion_features.squeeze(0).transpose(0, 1)

        return fusion_features