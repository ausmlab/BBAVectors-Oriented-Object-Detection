import torch
from torch import nn
from torch.nn import functional as F
from . import vgg


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y_ = y.view(b, c, 1, 1)

        # old version
        # P = x * y_.expand_as(x)
        # M = torch.mean(P, dim=1, keepdims=True)
        # P = F.avg_pool2d(P, (h, w))  # b,c,1,1

        # paper version
        # M=torch.sum(x * y_.expand_as(x),dim=1,keepdim=True)#b,1,h,w
        # # M=torch.sigmoid(M)
        # M=M/M.sum(dim=(-2,-1),keepdim=True)#b,1,h,w
        # P=x*M.expand_as(x)#b,1,h,w
        # P=F.avg_pool2d(P,(h,w))#b,c,1,1

        # paper revised version
        M = torch.sum(x * y_.expand_as(x), dim=1, keepdim=True)  # b,1,h,w
        M = F.normalize(M.view(b, -1), dim=-1, p=2).view(b, 1, h, w)
        P = x * M.expand_as(x)  # b,1,h,w
        P = F.avg_pool2d(P, (h, w))  # b,c,1,1

        return P, M.squeeze(dim=1), y


class MACNN(nn.Module):
    def __init__(self):
        super(MACNN, self).__init__()
        self.num_of_heads = 3
        self.vgg = vgg.vgg19(True)
        self.feat_dims = 512
        self.se1 = SELayer(self.feat_dims)
        self.se2 = SELayer(self.feat_dims)
        self.se3 = SELayer(self.feat_dims)
        self.se4 = SELayer(self.feat_dims)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cnnfc = nn.Linear(self.feat_dims, self.num_of_heads)

        self.fc1 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fc2 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fc3 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fc4 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fcall = nn.Linear(5 * self.feat_dims, self.num_of_heads)

    def forward(self, x):
        feat_maps = self.vgg(x)

        cnn_pred = self.cnnfc(self.pool(feat_maps).flatten(1))

        P1, M1, y1 = self.se1(feat_maps.detach())
        P2, M2, y2 = self.se2(feat_maps.detach())
        P3, M3, y3 = self.se3(feat_maps.detach())
        P4, M4, y4 = self.se4(feat_maps.detach())

        pred1 = self.fc1(P1.flatten(1))
        pred2 = self.fc2(P2.flatten(1))
        pred3 = self.fc3(P3.flatten(1))
        pred4 = self.fc4(P4.flatten(1))
        P = torch.cat([P1, P2, P3, P4, self.pool(feat_maps)], dim=1)
        pred = self.fcall(P.flatten(1))

        return feat_maps, cnn_pred, \
               [P1, P2, P3, P4], \
               [M1, M2, M3, M4], \
               [y1, y2, y3, y4], \
               [pred1, pred2, pred3, pred4, pred]


class MACNN_PART(nn.Module):
    def __init__(self):
        super(MACNN_PART, self).__init__()
        self.num_of_heads = 3
        self.vgg = vgg.vgg19(True)
        self.feat_dims = 512
        self.se1 = SELayer(self.feat_dims)
        self.se2 = SELayer(self.feat_dims)
        self.se3 = SELayer(self.feat_dims)
        self.se4 = SELayer(self.feat_dims)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cnnfc = nn.Linear(self.feat_dims, self.num_of_heads)
        self.cnnfcall = nn.Linear(self.num_of_heads * 4, self.num_of_heads)

        self.fc1 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fc2 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fc3 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fc4 = nn.Linear(self.feat_dims, self.num_of_heads)
        self.fcall = nn.Linear(8 * self.feat_dims, self.num_of_heads)

    def forward(self, x):
        top, middle, bottom, overall = x
        feat_maps_top = self.vgg(top)
        feat_maps_middle = self.vgg(middle)
        feat_maps_bottom = self.vgg(bottom)
        feat_maps_overall = self.vgg(overall)

        feat_maps = torch.cat([feat_maps_top, feat_maps_middle, feat_maps_bottom, feat_maps_overall], 1)
        cnn_pred = self.cnnfc(self.pool(feat_maps).flatten(1))

        P1, M1, y1 = self.se1(feat_maps_top.detach())
        P2, M2, y2 = self.se2(feat_maps_middle.detach())
        P3, M3, y3 = self.se3(feat_maps_bottom.detach())
        P4, M4, y4 = self.se4(feat_maps_overall.detach())

        pred1 = self.fc1(P1.flatten(1))
        pred2 = self.fc2(P2.flatten(1))
        pred3 = self.fc3(P3.flatten(1))
        pred4 = self.fc4(P4.flatten(1))
        P = torch.cat([P1, P2, P3, P4,
                       self.pool(feat_maps_top), self.pool(feat_maps_middle),
                       self.pool(feat_maps_bottom), self.pool(feat_maps_overall)],
                      dim=1)
        pred = self.fcall(P.flatten(1))

        return feat_maps, cnn_pred, \
               [P1, P2, P3, P4], \
               [M1, M2, M3, M4], \
               [y1, y2, y3, y4], \
               [pred1, pred2, pred3, pred4, pred]
