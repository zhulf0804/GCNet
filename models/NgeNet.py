import numpy as np
import os
import torch
import torch.nn as nn

from models.KPConv import block_decider
from models import InformationInteractive


class NgeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        r = config.first_subsampling_dl * config.conv_radius
        in_dim, out_dim = config.in_feats_dim, config.first_feats_dim
        K = config.num_kernel_points

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skips = [] # record the index of layers to be needed in decoder layer.
        self.encoder_skip_dims = [] # record the dims before pooling or strided-conv.
        block_i, layer_ind = 0, 0
        for block in config.architecture:
            if 'upsample' in block:
                break
            if np.any([skip_block in block for skip_block in ['strided', 'pool']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
            self.encoder_blocks.append(block_decider(block_name=block,
                                                     radius=r,
                                                     in_dim=in_dim,
                                                     out_dim=out_dim,
                                                     use_bn=config.use_batch_norm,
                                                     bn_momentum=config.batch_norm_momentum,
                                                     layer_ind=layer_ind,
                                                     config=config))

            in_dim = out_dim // 2 if 'simple' in block else out_dim
            if np.any([skip_block in block for skip_block in ['strided', 'pool']]):
                r *= 2
                out_dim *= 2
                layer_ind += 1
            block_i += 1

        # bottleneck layer
        self.bottleneck = nn.Conv1d(out_dim, config.gnn_feats_dim, 1)

        # Information Interactive block
        self.info_interative = InformationInteractive(layer_names=config.nets,
                                                      feat_dims=config.gnn_feats_dim,
                                                      gcn_k=config.dgcnn_k, 
                                                      ppf_k=config.ppf_k, 
                                                      radius=config.first_subsampling_dl*config.radius_mul,
                                                      bottleneck=config.bottleneck,
                                                      nhead=config.num_head)
        self.pro_gnn = nn.Conv1d(config.gnn_feats_dim, config.gnn_feats_dim, 1)
        self.attn_score = nn.Conv1d(config.gnn_feats_dim, 1, 1)
        self.epsilon = nn.Parameter(torch.tensor(-5.0))  # how to set ?

        # Decoder blocks
        out_dim = config.gnn_feats_dim + 2
        self.decoder_blocks = nn.ModuleList()
        self.decoder_skips = []
        layer = len(self.encoder_skip_dims) - 1

        self.decoder_blocks_m = nn.ModuleList()
        self.decoder_blocks_l = nn.ModuleList()
        cnt_upsample, mid_flag, low_flag = 0, True, True
        for block in config.architecture[block_i:]:
            if 'upsample' in block:
                layer_ind -= 1
                self.decoder_skips.append(block_i + 1)

            self.decoder_blocks.append(block_decider(block_name=block,
                                                     radius=r,
                                                     in_dim=in_dim,    # how to set for the first loop
                                                     out_dim=out_dim,
                                                     use_bn=config.use_batch_norm,
                                                     bn_momentum=config.batch_norm_momentum,
                                                     layer_ind=layer_ind,
                                                     config=config))

            if cnt_upsample >= 1:
                if cnt_upsample == 1 and mid_flag:
                    in_dim_clean = self.encoder_skip_dims[layer+1]
                    mid_flag = False
                else:
                    in_dim_clean = in_dim
                
                out_dim_clean = -1 if block == 'last_unary' else out_dim

                self.decoder_blocks_m.append(block_decider(block_name=block,
                                                           radius=r,
                                                           in_dim=in_dim_clean,    # how to set for the first loop
                                                           out_dim=out_dim_clean,
                                                           use_bn=config.use_batch_norm,
                                                           bn_momentum=config.batch_norm_momentum,
                                                           layer_ind=layer_ind,
                                                           config=config))

            if cnt_upsample >= 2:
                if cnt_upsample == 2 and low_flag:
                    in_dim_clean = self.encoder_skip_dims[layer+1] 
                    low_flag = False
                else: 
                    in_dim_clean = in_dim
                out_dim_clean = -1 if block == 'last_unary' else out_dim
                self.decoder_blocks_l.append(block_decider(block_name=block,
                                                           radius=r,
                                                           in_dim=in_dim_clean,    # how to set for the first loop
                                                           out_dim=out_dim_clean,
                                                           use_bn=config.use_batch_norm,
                                                           bn_momentum=config.batch_norm_momentum,
                                                           layer_ind=layer_ind,
                                                           config=config))

            in_dim = out_dim

            if 'upsample' in block:
                r *= 0.5
                in_dim += self.encoder_skip_dims[layer]
                layer -= 1
                out_dim = out_dim // 2
                cnt_upsample += 1

            block_i += 1

        # self.decoder_blocks_m = ['unary', 'nearest_upsample', 'unary', 'nearest_upsample', 'last_unary']
        # self.decoder_blocks_l = ['unary', 'nearest_upsample', 'last_unary']

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        stack_points = inputs['points']
        stacked_normals = inputs['normals']
        # stack_neighbors = inputs['neighbors']
        # stack_pools = inputs['pools']
        # stack_upsamples = inputs['upsamples']
        stack_lengths = inputs['stacked_lengths']
        # batched_coors = inputs['coors']
        # batched_transf = inputs['transf']

        # 1. encoder
        batched_feats = inputs['feats']
        block_i = 0
        skip_feats = []
        for block in self.encoder_blocks:
            if block_i in self.encoder_skips:
                skip_feats.append(batched_feats)
            batched_feats = block(batched_feats, inputs)
            block_i += 1

        # 2.1 bottleneck layer
        batched_feats = self.bottleneck(batched_feats.transpose(0, 1).unsqueeze(0)) # (1, gnn_feats_dim, n)

        # 2.2 information interaction
        len_src, len_tgt = stack_lengths[-1][0], stack_lengths[-1][1]
        coords_src, coords_tgt = stack_points[-1][:len_src], stack_points[-1][len_src:]
        coords_src, coords_tgt = coords_src.transpose(0, 1).unsqueeze(0), \
                                 coords_tgt.transpose(0, 1).unsqueeze(0)
        feats_src, feats_tgt = batched_feats[:, :, :len_src], \
                               batched_feats[:, :, len_src:]
        normals_src = stacked_normals[-1][:len_src].transpose(0, 1).unsqueeze(0)
        normals_tgt = stacked_normals[-1][len_src:].transpose(0, 1).unsqueeze(0)
        feats_src, feats_tgt = self.info_interative(coords_src, feats_src, coords_tgt, feats_tgt, normals_src, normals_tgt)
        batched_feats = torch.cat([feats_src, feats_tgt], dim=-1)
        batched_feats = self.pro_gnn(batched_feats) # why this one ?

        # 2.3 overlap score
        attn_scores = self.attn_score(batched_feats).squeeze(0).transpose(0, 1) # (n, 1)
        temperature = torch.exp(self.epsilon) + 0.03
        batched_feats_norm = batched_feats / (torch.norm(batched_feats, dim=1, keepdim=True) + 1e-8)
        batched_feats_norm = batched_feats_norm.squeeze(0).transpose(0, 1) # (n, c)
        feats_norm_src, feats_norm_tgt = batched_feats_norm[:len_src], \
                                         batched_feats_norm[len_src:]
        inner_product = torch.matmul(feats_norm_src, feats_norm_tgt.transpose(0, 1)) # (n1, n2), n1 + n2
        attn_scores_src, attn_scores_tgt = attn_scores[:len_src], attn_scores[len_src:] # (n1, 1), (n2, 1)
        ol_scores_src = torch.matmul(torch.softmax(inner_product / temperature, dim=1), attn_scores_tgt) # (n1, 1)
        ol_scores_tgt = torch.matmul(torch.softmax(inner_product.transpose(0, 1) / temperature, dim=1), attn_scores_src) # (n2, 1)
        ol_scores = torch.cat([ol_scores_src, ol_scores_tgt], dim=0) # (n, 1)

        # 2.4 feats
        batched_feats_raw = batched_feats.squeeze(0).transpose(0, 1)  # (n, c)
        batched_feats = torch.cat([batched_feats_raw, attn_scores, ol_scores], dim=1)

        # 3. decoder
        cnt_decoder = 0
        for ind, block in enumerate(self.decoder_blocks):
            if block_i in self.decoder_skips:
                cnt_decoder += 1
                cur_skip_feats = skip_feats.pop()
                batched_feats = torch.cat([batched_feats, cur_skip_feats], dim=-1)
                if cnt_decoder >= 1:
                    if cnt_decoder == 1:
                        batched_feats_m = cur_skip_feats
                    else:
                        batched_feats_m = torch.cat([batched_feats_m, cur_skip_feats], dim=-1)
                if cnt_decoder >= 2:
                    if cnt_decoder == 2:
                        batched_feats_l = cur_skip_feats
                    else:
                        batched_feats_l = torch.cat([batched_feats_l, cur_skip_feats], dim=-1)
            
            if cnt_decoder >= 1:
                block_m = self.decoder_blocks_m[ind - 1]
                batched_feats_m = block_m(batched_feats_m, inputs)
            
            if cnt_decoder >= 2:
                block_l = self.decoder_blocks_l[ind - (self.decoder_skips[1] - self.decoder_skips[0] + 1)]
                batched_feats_l = block_l(batched_feats_l, inputs)
            
            batched_feats = block(batched_feats, inputs)
            block_i += 1
        
        overlap_scores = self.sigmoid(batched_feats[:, -2:-1])
        saliency_scores = self.sigmoid(batched_feats[:, -1:])
        batched_feats = batched_feats[:, :-2] / torch.norm(batched_feats[:, :-2], dim=1, keepdim=True)
        batched_feats_m = batched_feats_m / torch.norm(batched_feats_m, dim=1, keepdim=True)
        batched_feats_l = batched_feats_l / torch.norm(batched_feats_l, dim=1, keepdim=True)
        batched_feats = torch.cat([batched_feats, overlap_scores, saliency_scores], dim=-1)

        return batched_feats, batched_feats_m, batched_feats_l