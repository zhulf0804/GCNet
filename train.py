import numpy as np
import os
import shutil
import sys
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset, get_dataloader
from models import architectures, NgeNet
from losses import Loss
from utils import decode_config, setup_seed

CUR = os.path.dirname(os.path.abspath(__file__))


def save_summary(writer, loss_dict, global_step, tag, lr=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)


def main():
    setup_seed(1234)
    config = decode_config(sys.argv[1])
    config = edict(config)
    config.architecture = architectures[config.dataset]

    saved_path = config.exp_dir
    saved_ckpt_path = os.path.join(saved_path, 'checkpoints')
    saved_logs_path = os.path.join(saved_path, 'summary')
    os.makedirs(saved_path, exist_ok=True)
    os.makedirs(saved_ckpt_path, exist_ok=True)
    os.makedirs(saved_logs_path, exist_ok=True)
    shutil.copyfile(sys.argv[1], os.path.join(saved_path, f'{config.dataset}.yaml'))
    train_dataset, val_dataset = get_dataset(config.dataset, config)
    train_dataloader, neighborhood_limits = get_dataloader(config=config,
                                                           dataset=train_dataset,
                                                           batch_size=config.batch_size,
                                                           num_workers=config.num_workers,
                                                           shuffle=True,
                                                           neighborhood_limits=None)
    val_dataloader, _ = get_dataloader(config=config,
                                       dataset=val_dataset,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       neighborhood_limits=neighborhood_limits)

    print(neighborhood_limits)
    model = NgeNet(config).cuda()
    model_loss = Loss(config)

    if config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )

    # create learning rate scheduler
    if config.scheduler == 'ExpLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler_gamma,
        )
    elif config.scheduler == 'CosA':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=config.T_0,
                                                                         T_mult=config.T_mult,
                                                                         eta_min=config.eta_min,
                                                                         last_epoch=-1)
    else:
        raise ValueError


    writer = SummaryWriter(saved_logs_path)

    best_recall, best_recall_sum, best_circle_loss, best_loss = 0, 0, 1e8, 1e8
    w_saliency = config.w_saliency_loss
    w_saliency_update = False

    for epoch in range(config.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for inputs in tqdm(train_dataloader):
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda()
                else:
                    inputs[k] = inputs[k].cuda()

            optimizer.zero_grad()

            batched_feats, batched_feats_m, batched_feats_l = model(inputs)
            stack_points = inputs['points']
            stack_lengths = inputs['stacked_lengths']
            feats_src = batched_feats[:stack_lengths[0][0]]
            feats_tgt = batched_feats[stack_lengths[0][0]:]
            feats_src_m = batched_feats_m[:stack_lengths[0][0]]
            feats_tgt_m = batched_feats_m[stack_lengths[0][0]:]
            feats_src_l = batched_feats_l[:stack_lengths[0][0]]
            feats_tgt_l = batched_feats_l[stack_lengths[0][0]:]

            coors = inputs['coors'][0] # list, [coors1, coors2, ..], preparation for batchsize > 1
            transf = inputs['transf'][0] # (1, 4, 4), preparation for batchsize > 1
            points_raw = inputs['batched_points_raw']
            coords_src = points_raw[:stack_lengths[0][0]]
            coords_tgt = points_raw[stack_lengths[0][0]:]

            loss_dict = model_loss(coords_src=coords_src,
                                   coords_tgt=coords_tgt,
                                   feats_src=feats_src,
                                   feats_tgt=feats_tgt,
                                   feats_src_m=feats_src_m,
                                   feats_tgt_m=feats_tgt_m,
                                   feats_src_l=feats_src_l,
                                   feats_tgt_l=feats_tgt_l,
                                   coors=coors,
                                   transf=transf,
                                   w_saliency=w_saliency)

            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % config.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'])
            train_step += 1

            # This line of code reduces the training speed. 
            # If GPU memory allows, it is recommended not to add this line of code or add this line after each epoch
            torch.cuda.empty_cache()
        scheduler.step()

        total_circle_loss, total_recall, total_loss, total_recall_sum = [], [], [], []
        model.eval()
        with torch.no_grad():
            for inputs in tqdm(val_dataloader):
                for k, v in inputs.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            inputs[k][i] = inputs[k][i].cuda()
                    else:
                        inputs[k] = inputs[k].cuda()

                batched_feats, batched_feats_m, batched_feats_l = model(inputs)
                stack_points = inputs['points']
                stack_lengths = inputs['stacked_lengths']
                feats_src = batched_feats[:stack_lengths[0][0]]
                feats_tgt = batched_feats[stack_lengths[0][0]:]
                feats_src_m = batched_feats_m[:stack_lengths[0][0]]
                feats_tgt_m = batched_feats_m[stack_lengths[0][0]:]
                feats_src_l = batched_feats_l[:stack_lengths[0][0]]
                feats_tgt_l = batched_feats_l[stack_lengths[0][0]:]

                coors = inputs['coors'][0] # list, [coors1, coors2, ..], preparation for batchsize > 1
                transf = inputs['transf'][0] # (1, 4, 4), preparation for batchsize > 1
                points_raw = inputs['batched_points_raw']
                coords_src = points_raw[:stack_lengths[0][0]]
                coords_tgt = points_raw[stack_lengths[0][0]:]

                loss_dict = model_loss(coords_src=coords_src,
                                       coords_tgt=coords_tgt,
                                       feats_src=feats_src,
                                       feats_tgt=feats_tgt,
                                       feats_src_m=feats_src_m,
                                       feats_tgt_m=feats_tgt_m,
                                       feats_src_l=feats_src_l,
                                       feats_tgt_l=feats_tgt_l,
                                       coors=coors,
                                       transf=transf,
                                       w_saliency=w_saliency)

                loss = loss_dict['circle_loss'] + loss_dict['circle_loss_m'] + loss_dict['circle_loss_l']
                total_loss.append(loss.detach().cpu().numpy())
                circle_loss = loss_dict['circle_loss']
                total_circle_loss.append(circle_loss.detach().cpu().numpy())
                recall = loss_dict['recall']
                total_recall.append(recall.detach().cpu().numpy())
                recall_sum = loss_dict['recall'] + loss_dict['recall_m'] + loss_dict['recall_l']
                total_recall_sum.append(recall_sum.detach().cpu().numpy())

                global_step = epoch * len(val_dataloader) + val_step + 1

                if global_step % config.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1
                
                # This line of code reduces the training speed. 
                # If GPU memory allows, it is recommended not to add this line of code or add this line after each epoch
                torch.cuda.empty_cache()

        if np.mean(total_circle_loss) < best_circle_loss:
            best_circle_loss = np.mean(total_circle_loss)
            torch.save(model.state_dict(), os.path.join(saved_ckpt_path, 'best_loss.pth'))
        if np.mean(total_recall) > best_recall:
            best_recall = np.mean(total_recall)
            torch.save(model.state_dict(), os.path.join(saved_ckpt_path, 'best_recall.pth'))
        if not w_saliency_update and np.mean(total_recall) > 0.3:
            w_saliency_update = True
            w_saliency = 1

        model.train()


if __name__ == '__main__':
    main()
