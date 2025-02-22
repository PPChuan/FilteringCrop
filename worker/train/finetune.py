import os
import sys
import time

from torch import autocast

sys.path.append(os.path.abspath('../../'))
from F_Crop.build.util import *


def finetune(model, train_loader, val_loader, optimizer, criterion, cfg, epoch):
    acc1_gather = 0
    acc5_gather = 0
    t_avg_time = 0
    t_avg_loss = 0
    v_avg_time = 0
    v_avg_loss = 0
    t_batch_count = len(train_loader)
    v_batch_count = len(val_loader)
    if cfg.local_rank == 0:
        tl = tqdm(train_loader, leave=False, desc="FineTune! epoch" + str(epoch), unit="batch")
        vl = tqdm(val_loader, leave=False, desc="Val! epoch" + str(epoch), unit="batch")
    else:
        tl = train_loader
        vl = val_loader
    model.eval()
    for images, labels in tl:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        t = time.time()
        preds = model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_avg_time += time.time() - t
        t_loss = torch.tensor(loss.item(), dtype=torch.float).cuda(cfg.local_rank)
        dist.all_reduce(t_loss)
        t_avg_loss += t_loss.item() / cfg.world_size

    with torch.no_grad():
        for images, labels in vl:
            val_images = images.float().cuda()
            val_labels = labels.cuda()

            t = time.time()
            val_preds = model(val_images)
            v_avg_time += time.time() - t

            acc1 = torch.tensor(get_num_acc1(val_preds, val_labels), dtype=torch.float).cuda(cfg.local_rank)
            dist.all_reduce(acc1)
            acc1_gather += acc1.item()
            acc5 = torch.tensor(get_num_acc5(val_preds, val_labels), dtype=torch.float).cuda(cfg.local_rank)
            dist.all_reduce(acc5)
            acc5_gather += acc5.item()
            val_loss = criterion(val_preds, val_labels)
            v_loss = torch.tensor(val_loss.item(), dtype=torch.float).cuda(cfg.local_rank)
            dist.all_reduce(v_loss)
            v_avg_loss += v_loss.item() / cfg.world_size

    return t_avg_loss / t_batch_count, v_avg_loss / v_batch_count, t_avg_time / t_batch_count, v_avg_time / v_batch_count, acc1_gather, acc5_gather


@torch.no_grad()
def get_num_acc1(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


@torch.no_grad()
def get_num_acc5(preds, labels):
    value, indices = torch.topk(preds, 5)
    return torch.transpose(indices.reshape(-1, 5), 0, 1).eq(labels.repeat(1, 5).reshape(5, -1)).sum().item()


def finetune_amp(model, train_loader, val_loader, optimizer, criterion, cfg, epoch, scaler):
    acc1_gather = 0
    acc5_gather = 0
    t_avg_time = 0
    t_avg_loss = 0
    v_avg_time = 0
    v_avg_loss = 0
    t_batch_count = len(train_loader)
    v_batch_count = len(val_loader)
    if cfg.local_rank == 0:
        tl = tqdm(train_loader, leave=False, desc="FineTune! epoch" + str(epoch), unit="batch")
        vl = tqdm(val_loader, leave=False, desc="Val! epoch" + str(epoch), unit="batch")
    else:
        tl = train_loader
        vl = val_loader
    model.eval()
    for images, labels in tl:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        t = time.time()
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = model(images)
            loss = criterion(preds, labels)

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        t_avg_time += time.time() - t
        t_loss = torch.tensor(loss.item(), dtype=torch.float).cuda(cfg.local_rank)
        dist.all_reduce(t_loss)
        t_avg_loss += t_loss.item() / cfg.world_size

    with torch.no_grad():
        for images, labels in vl:
            val_images = images.float().cuda()
            val_labels = labels.cuda()

            t = time.time()
            val_preds = model(val_images)
            v_avg_time += time.time() - t

            acc1 = torch.tensor(get_num_acc1(val_preds, val_labels), dtype=torch.float).cuda(cfg.local_rank)
            dist.all_reduce(acc1)
            acc1_gather += acc1.item()
            acc5 = torch.tensor(get_num_acc5(val_preds, val_labels), dtype=torch.float).cuda(cfg.local_rank)
            dist.all_reduce(acc5)
            acc5_gather += acc5.item()
            val_loss = criterion(val_preds, val_labels)
            v_loss = torch.tensor(val_loss.item(), dtype=torch.float).cuda(cfg.local_rank)
            dist.all_reduce(v_loss)
            v_avg_loss += v_loss.item() / cfg.world_size

    return t_avg_loss / t_batch_count, v_avg_loss / v_batch_count, t_avg_time / t_batch_count, v_avg_time / v_batch_count, acc1_gather, acc5_gather
