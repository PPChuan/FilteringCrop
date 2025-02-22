import os
import sys
import time

from torch import autocast

sys.path.append(os.path.abspath('../../'))
from F_Crop.build.util import *


def pretrain(model, dataloader, optimizer, criterion, cfg, epoch):
    avg_time = 0
    avg_loss = 0
    batch_count = len(dataloader)

    model.train()
    for images, labels in tqdm(dataloader, leave=False, desc="Pretrain! epoch" + str(epoch),
                               unit="batch") if cfg.local_rank == 0 else dataloader:
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        t = time.time()

        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_time += time.time() - t
        m_loss = torch.tensor(loss.item(), dtype=torch.float).cuda(cfg.local_rank)

        dist.all_reduce(m_loss)

        avg_loss += m_loss.item() / cfg.world_size

    return avg_loss / batch_count, avg_time / batch_count


def pretrain_amp_bf16(model, dataloader, optimizer, criterion, cfg, epoch, scaler):
    avg_time = 0
    avg_loss = 0
    batch_count = len(dataloader)

    model.train()
    for images, labels in tqdm(dataloader, leave=False, desc="Pretrain! epoch" + str(epoch),
                               unit="batch") if cfg.local_rank == 0 else dataloader:
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        t = time.time()
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        avg_time += time.time() - t
        m_loss = torch.tensor(loss.item(), dtype=torch.float).cuda(cfg.local_rank)

        dist.all_reduce(m_loss)

        avg_loss += m_loss.item() / cfg.world_size

    return avg_loss / batch_count, avg_time / batch_count


def pretrain_amp_f16(model, dataloader, optimizer, criterion, cfg, epoch, scaler):
    avg_time = 0
    avg_loss = 0
    batch_count = len(dataloader)

    model.train()
    for images, labels in tqdm(dataloader, leave=False, desc="Pretrain! epoch" + str(epoch),
                               unit="batch") if cfg.local_rank == 0 else dataloader:
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        t = time.time()
        with autocast(device_type='cuda', dtype=torch.float16):
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        avg_time += time.time() - t
        m_loss = torch.tensor(loss.item(), dtype=torch.float).cuda(cfg.local_rank)

        dist.all_reduce(m_loss)

        avg_loss += m_loss.item() / cfg.world_size

    return avg_loss / batch_count, avg_time / batch_count
