import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from models.KPConv import load_kernels


# very important for backward speed !!
def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


class KPConv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, K, KP_extent,
                 KP_influence='linear', aggregation_mode='sum'):
        '''
        Basic KPConv implementation without deformation.
        :param in_channels: int,
        :param out_channels: int,
        :param radius: float,
        :param K: int, number of kernel points
        :param KP_extent: float,
        :param KP_influence: str, ['linear', 'constant']
        :param aggregation_mode: str, ['sum', 'cloest']
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.K = K
        self.KP_extent = KP_extent
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode

        self.weights = Parameter(torch.zeros((K, in_channels, out_channels),
                                             dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.kernel_points = nn.Parameter(torch.tensor(load_kernels(radius, K),
                                                       dtype=torch.float32),
                                          requires_grad=False) # (K, 3)

    def forward(self, q_pts, s_pts, s_feats, neigh_inds):
        '''

        :param q_pts: (n, 3)
        :param s_pts: (m, 3)
        :param s_feats: (m, c_in)
        :param neigh_inds: (n, k)
        :return: q_feats: (n, c_out)
        '''
        # a trick for radius query results.
        s_pts = torch.cat([s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6], dim=0)
        neigh_pts = s_pts[neigh_inds] # (n, k, 3)
        s_feats = torch.cat([s_feats, torch.zeros_like(s_feats[:1, :])], dim=0)
        # neigh_feats = torch.unsqueeze(s_feats[neigh_inds], dim=2) # (n, k, 1, c_in)
        # neigh_feats = s_feats[neigh_inds] # (n, k, c_in)
        neigh_feats = gather(s_feats, neigh_inds)

        # center every neighborhood
        neigh_pts = neigh_pts - torch.unsqueeze(q_pts, dim=1)
        neigh_pts.unsqueeze_(2) # (n, k, 1, 3)
        differences = neigh_pts - self.kernel_points # (n, k, K, 3)
        sq_distance = torch.sum(torch.pow(differences, 2), dim=-1) + 1e-8 # (n, k, K)
        if self.KP_influence == 'constant':
            h = torch.ones_like(sq_distance)
        elif self.KP_influence == 'linear':
            h = torch.clamp(1 - torch.sqrt(sq_distance) / self.KP_extent, min=0) # (n, k, K)
        else:
            raise NotImplementedError

        if self.aggregation_mode == 'cloest':
            neighbors_1nn = torch.argmin(sq_distance, dim=2) # (n, k)
            mask = torch.nn.functional.one_hot(neighbors_1nn, num_classes=self.K) # (n, k, K)
            h *= mask
        elif self.aggregation_mode == 'sum':
            pass
        else:
            raise NotImplementedError

        '''
        # The following explict implementation causes "out of memory"
        kernel_g = torch.einsum('nke, eio->nkio', h, self.weights) # (n, k, c_in, c_out)
        q_feats_local = torch.squeeze(torch.matmul(neigh_feats, kernel_g), dim=2) # (n, k, c_out)
        q_feats = torch.sum(q_feats_local, dim=1)
        return q_feats # (n, c_out)
        '''
        h = torch.transpose(h, 1, 2) # (n, K, k)
        kernel_feats = torch.matmul(h, neigh_feats) # (n, K, c_in)
        kernel_feats = torch.transpose(kernel_feats, 0, 1)
        q_feats_local = torch.matmul(kernel_feats, self.weights) # (K, n, c_out)
        q_feats = torch.sum(q_feats_local, dim=0) # (n, c_out)

        # normalization, is it necessary ? torch.gt() is good ?
        neighbor_num = torch.sum(torch.gt(torch.sum(neigh_feats, dim=-1), 0), dim=-1)
        neighbor_num = torch.clamp(neighbor_num, min=1).unsqueeze(1)
        normalized_q_feats = q_feats / neighbor_num
        return normalized_q_feats

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, nkernels: {:d}, in_feat: {:d}, out_feat: {:d})'.\
            format(self.radius, self.K, self.in_channels, self.out_channels)


class BatchNormBlock(nn.Module):
    def __init__(self, in_dim, use_bn, bn_momentum):
        super().__init__()
        self.use_bn = use_bn
        if self.use_bn:
            # self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, features):
        '''

        :param features: (n, c)
        :return:
        '''
        if self.use_bn:
            features = torch.unsqueeze(features, 0).permute(0, 2, 1).contiguous()
            features = self.batch_norm(features)
            features = torch.squeeze(features.permute(0, 2, 1), 0).contiguous()
        else:
            features += self.bias
        return features


class UnaryBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, relu=True):
        super().__init__()
        self.relu = relu
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = BatchNormBlock(out_dim, use_bn, bn_momentum)
        if self.relu:
            self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, features, batch=None):
        # batch=None is a trick, sync with nn.Identity()
        features = self.bn(self.mlp(features))
        if self.relu:
            return self.leaky_relu(features)
        return features


class MaxPoolBlock(nn.Module):
    def __init__(self, layer_ind):
        super().__init__()
        self.layer_ind = layer_ind

    def forward(self, feats, batch):
        # maybe we need to optimize gather later ?
        feats = torch.cat([feats, torch.zeros_like(feats[:1, :])], dim=0)
        pooled_inds = batch['pools'][self.layer_ind]
        # gatherd_feats = feats[pooled_inds, :]
        gatherd_feats = gather(feats, pooled_inds)
        return torch.max(gatherd_feats, dim=1)[0]


class NearestUpsampleBlock(nn.Module):
    def __init__(self, layer_ind):
        super().__init__()
        self.layer_ind = layer_ind

    def forward(self, feats, batch):
        neigh_inds = batch['upsamples'][self.layer_ind]
        # return feats[neigh_inds[:, 0], :]
        return gather(feats, neigh_inds[:, 0])


class SimpleBlock(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, use_bn, bn_momentum, layer_ind, config):
        super().__init__()
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius
        KP_extent = radius * config.KP_extent / config.conv_radius

        self.KPConv = KPConv(in_channels=in_dim,
                             out_channels=out_dim // 2,
                             radius=radius,
                             K=config.num_kernel_points,
                             KP_extent=KP_extent,
                             KP_influence='linear',
                             aggregation_mode='sum')
        self.bn = BatchNormBlock(out_dim // 2, use_bn, bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, batch):
        s_pts = batch['points'][self.layer_ind]
        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            neigh_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            neigh_inds = batch['neighbors'][self.layer_ind]
        q_feats = self.KPConv(q_pts, s_pts, s_feats, neigh_inds)
        q_feats = self.bn(q_feats)
        q_feats = self.leaky_relu(q_feats)
        return q_feats


class ResnetBottleneckBlock(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, use_bn, bn_momentum, layer_ind, config):
        super().__init__()
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius
        self.layer_ind = layer_ind
        KP_extent = radius * config.KP_extent / config.conv_radius

        # dimensionality reduction
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, use_bn, bn_momentum)
        else:
            self.unary1 = nn.Identity()
        self.KPConv = KPConv(in_channels=out_dim // 4,
                             out_channels=out_dim // 4,
                             radius=radius,
                             K=config.num_kernel_points,
                             KP_extent=KP_extent,
                             KP_influence='linear',
                             aggregation_mode='sum')
        self.bn = BatchNormBlock(out_dim // 4, use_bn, bn_momentum)
        # dimensionality increase
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, use_bn, bn_momentum, relu=False) # why no relu ?

        self.max_pool = MaxPoolBlock(layer_ind=layer_ind)
        # shortcut layer
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, use_bn, bn_momentum, relu=False) # why no relu ?
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, batch):
        s_pts = batch['points'][self.layer_ind]
        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            neigh_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            neigh_inds = batch['neighbors'][self.layer_ind]

        s_feats_d = self.unary1(s_feats)
        q_feats = self.KPConv(q_pts, s_pts, s_feats_d, neigh_inds)
        q_feats = self.leaky_relu(self.bn(q_feats))
        q_feats = self.unary2(q_feats)

        if 'strided' in self.block_name:
            shortcut = self.max_pool(s_feats, batch)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)
        return self.leaky_relu(q_feats + shortcut)


def block_decider(block_name, radius, in_dim, out_dim, use_bn, bn_momentum, layer_ind, config):
    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, use_bn, bn_momentum)

    if block_name == 'last_unary':
        v = 0 if out_dim == -1 else 2
        return UnaryBlock(in_dim,
                          config.final_feats_dim+v,
                          use_bn=False,
                          bn_momentum=bn_momentum,
                          relu=False)

    if block_name in ['simple']:
        return SimpleBlock(block_name=block_name,
                           in_dim=in_dim,
                           out_dim=out_dim,
                           radius=radius,
                           use_bn=use_bn,
                           bn_momentum=bn_momentum,
                           layer_ind=layer_ind,
                           config=config)

    if block_name in ['resnetb', 'resnetb_strided']:
        return ResnetBottleneckBlock(block_name=block_name,
                                     in_dim=in_dim,
                                     out_dim=out_dim,
                                     radius=radius,
                                     use_bn=use_bn,
                                     bn_momentum=bn_momentum,
                                     layer_ind=layer_ind,
                                     config=config)

    if block_name in ['nearest_upsample']:
        return NearestUpsampleBlock(layer_ind=layer_ind)
    raise NotImplementedError


if __name__ == '__main__':
    import numpy as np
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    q_pts = torch.rand(2, 3)
    s_pts = torch.randn(5, 3)
    s_feats = torch.rand(5, 4)
    neigh_inds = torch.tensor([[1, 2], [0, 3]])

    conv = KPConv(4, 8, 2, 15, 2)
    q_feats = conv(q_pts, s_pts, s_feats, neigh_inds)
    print(q_feats)
    print(conv)
