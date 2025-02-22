import os
import sys
import time

from torch import nn

sys.path.append(os.path.abspath('../../'))
from F_Crop.build.util import *


def pretrain(model, dataloader, optimizer, criterion, cfg, epoch):
    avg_time = 0
    avg_loss = 0
    batch_count = len(dataloader)

    model.train()
    for images, labels in tqdm(dataloader, leave=False, desc="Pretrain! epoch" + str(epoch),
                               unit="batch") if cfg.local_rank == 0 else dataloader:
        bsz = images[0].shape[0]

        if epoch <= cfg.init_epoch:
            images[1] = torch.randn(images[0].shape)
            images[1] = nn.functional.normalize(images[1], dim=0)

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda(cfg.local_rank, non_blocking=True)

        t = time.time()

        features = model(images)  # (2*bsz, C)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        loss = criterion(f1, f2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_time += time.time() - t
        m_loss = torch.tensor(loss.item(), dtype=torch.float).cuda(cfg.local_rank)

        dist.all_reduce(m_loss)

        avg_loss += m_loss.item() / cfg.world_size

    return avg_loss / batch_count, avg_time / batch_count
