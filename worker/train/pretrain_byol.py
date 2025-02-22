import sys
import time
import os

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

        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -2 * (criterion(p1, z2).mean() + criterion(p2, z1).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_time += time.time() - t
        m_loss = torch.tensor(loss.item(), dtype=torch.float).cuda(cfg.local_rank)

        dist.all_reduce(m_loss)

        avg_loss += m_loss.item() / cfg.world_size

    return avg_loss / batch_count, avg_time / batch_count
