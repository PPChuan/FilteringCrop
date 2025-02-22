import os
import sys

from torch.backends import cudnn

sys.path.append(os.path.abspath('../'))
from F_Crop.build import *
from .train.pretrain_byol import pretrain
from F_Crop.run_manager_ckp import *


def pretrain_worker_rcrop(rank, world_size, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.pretrain_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    pretrain_set = build_rcrop_datasets(local_args.dataset)
    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set, shuffle=True)
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=pretrain_sampler,
        drop_last=True
    )

    local_args.byol.encoder_q = build_model(local_args.model)
    local_args.byol.encoder_k = build_model(local_args.model)
    model = build_model(local_args.byol)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank],
                                                      find_unused_parameters=True)

    criterion = build_criterion(local_args.loss.pretrain).cpu()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    if local_args.args.resume is not None:
        checkpoint = torch.load(os.path.join('./', local_args.args.resume, 'ckp', 'ckp.pth'), map_location='cuda')
        model.load_state_dict(checkpoint['byol_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        local_args.start_epoch = checkpoint['epoch'] + 1
        print('start_epoch = ' + str(local_args.start_epoch))

        local_args.data = checkpoint['data_manager_param']['data']
        local_args.run_stime = checkpoint['data_manager_param']['run_stime']
        local_args.module_cost = checkpoint['data_manager_param']['module_cost']
        local_args.best_acc1 = checkpoint['data_manager_param']['best_acc1']
        local_args.best_acc5 = checkpoint['data_manager_param']['best_acc5']
        local_args.best_acc1_epoch = checkpoint['data_manager_param']['best_acc1_epoch']
        local_args.best_acc5_epoch = checkpoint['data_manager_param']['best_acc5_epoch']

    else:
        local_args.start_epoch = 1
        local_args.data = []
        local_args.run_stime = time.time()
        local_args.module_cost = 0
        local_args.best_acc1 = 0
        local_args.best_acc5 = 0
        local_args.best_acc1_epoch = 0
        local_args.best_acc5_epoch = 0

    cudnn.benchmark = True

    if local_args.local_rank == 0:
        dm = DataManager(local_args)

    for epoch in range(local_args.start_epoch, local_args.run_p.pretrain_epoch + 1):

        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])
        loss, batch_time = pretrain(model, pretrain_loader, optimizer, criterion, local_args, epoch)
        if local_args.local_rank == 0:
            if not local_args.ifTry:
                dm.save_ckp(model, optimizer, 'byol_state')
            dm.end_pretrain_epoch(model.module, loss, batch_time)


def pretrain_worker_ccrop(rank, world_size, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.pretrain_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    pretrain_set, eval_set = build_ccrop_datasets(local_args.dataset)
    len_ds = len(pretrain_set)

    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set, shuffle=True)
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=pretrain_sampler,
        drop_last=True
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=eval_sampler,
        drop_last=False
    )

    local_args.byol.encoder_q = build_model(local_args.model)
    local_args.byol.encoder_k = build_model(local_args.model)
    model = build_model(local_args.byol)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank],
                                                      find_unused_parameters=True)

    criterion = build_criterion(local_args.loss.pretrain).cuda()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    if local_args.args.resume is not None:
        checkpoint = torch.load(os.path.join('./', local_args.args.resume, 'ckp', 'ckp.pth'), map_location='cuda')
        model.load_state_dict(checkpoint['moco_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        local_args.start_epoch = checkpoint['epoch'] + 1
        print('start_epoch = ' + str(local_args.start_epoch))
        pretrain_set.boxes = checkpoint['boxes'].cpu()

        local_args.data = checkpoint['data_manager_param']['data']
        local_args.run_stime = checkpoint['data_manager_param']['run_stime']
        local_args.module_cost = checkpoint['data_manager_param']['module_cost']
        local_args.best_acc1 = checkpoint['data_manager_param']['best_acc1']
        local_args.best_acc5 = checkpoint['data_manager_param']['best_acc5']
        local_args.best_acc1_epoch = checkpoint['data_manager_param']['best_acc1_epoch']
        local_args.best_acc5_epoch = checkpoint['data_manager_param']['best_acc5_epoch']

    else:
        local_args.start_epoch = 1
        local_args.data = []
        local_args.run_stime = time.time()
        local_args.module_cost = 0
        local_args.best_acc1 = 0
        local_args.best_acc5 = 0
        local_args.best_acc1_epoch = 0
        local_args.best_acc5_epoch = 0

    cudnn.benchmark = True

    if local_args.local_rank == 0:
        dm = DataManager(local_args)

    for epoch in range(local_args.start_epoch, local_args.run_p.pretrain_epoch + 1):
        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])

        pretrain_set.use_box = epoch >= local_args.box.warmup_epochs + 1

        loss, batch_time = pretrain(model, pretrain_loader, optimizer, criterion, local_args, epoch)

        if epoch >= local_args.box.warmup_epochs and epoch != local_args.run_p.pretrain_epoch and (
                epoch - local_args.box.warmup_epochs) % local_args.box.loc_interval == 0:
            ts = time.time()
            # all_boxes: tensor (len_ds, 4); (h_min, w_min, h_max, w_max)
            all_boxes = update_box(eval_loader, model.module.encoder_q, len_ds,
                                   True if local_args.local_rank == 0 else False,
                                   t=local_args.box.box_thresh)  # on_cuda=True
            assert len(all_boxes) == len_ds
            pretrain_set.boxes = all_boxes.cpu()
            dm.acc_module_cost(time.time() - ts)

        if local_args.local_rank == 0:
            if not local_args.ifTry:
                dm.save_ckp(model, optimizer, 'byol_state', ['boxes', pretrain_set.boxes])
            dm.end_pretrain_epoch(model.module, loss, batch_time)


def pretrain_worker_Fcrop(rank, world_size, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.pretrain_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    pretrain_set, eval_set, init_set = build_Fcrop_datasets(local_args.dataset)
    len_ds = len(pretrain_set)

    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set, shuffle=True)
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=pretrain_sampler,
        drop_last=True
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=eval_sampler,
        drop_last=False
    )

    init_loader = torch.utils.data.DataLoader(
        init_set,
        batch_size=1,
        drop_last=False
    )

    local_args.byol.encoder_q = build_model(local_args.model)
    local_args.byol.encoder_k = build_model(local_args.model)
    model = build_model(local_args.byol)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank],
                                                      find_unused_parameters=True)

    criterion = build_criterion(local_args.loss.pretrain).cpu()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    if local_args.args.resume is not None:
        checkpoint = torch.load(os.path.join('./', local_args.args.resume, 'ckp', 'ckp.pth'), map_location='cuda')
        model.load_state_dict(checkpoint['byol_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        local_args.start_epoch = checkpoint['epoch'] + 1
        print('start_epoch = ' + str(local_args.start_epoch))
        pretrain_set.centers = checkpoint['centers'].cpu()

        init_time_cost = -1
        local_args.data = checkpoint['data_manager_param']['data']
        local_args.run_stime = checkpoint['data_manager_param']['run_stime']
        local_args.module_cost = checkpoint['data_manager_param']['module_cost']
        local_args.best_acc1 = checkpoint['data_manager_param']['best_acc1']
        local_args.best_acc5 = checkpoint['data_manager_param']['best_acc5']
        local_args.best_acc1_epoch = checkpoint['data_manager_param']['best_acc1_epoch']
        local_args.best_acc5_epoch = checkpoint['data_manager_param']['best_acc5_epoch']

    else:
        if os.path.exists(os.path.join('./', local_args.dataset.root, 'center_ckp.pth')):
            checkpoint = torch.load(os.path.join('./', local_args.dataset.root, 'center_ckp.pth'), map_location='cuda')
            all_centers = checkpoint['init_centers']
            init_time_cost = checkpoint['init_time_cost']
        else:
            all_centers, init_time_cost = get_Center(init_loader)
            save_centers(local_args.dataset.root, all_centers, init_time_cost)
        assert len(all_centers) == len_ds
        print("Init Time Consumption: " + s2hms(init_time_cost))
        pretrain_set.centers = all_centers.cpu()
        local_args.start_epoch = 1
        local_args.data = []
        local_args.run_stime = time.time()
        local_args.module_cost = 0
        local_args.best_acc1 = 0
        local_args.best_acc5 = 0
        local_args.best_acc1_epoch = 0
        local_args.best_acc5_epoch = 0

    cudnn.benchmark = True

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
        if init_time_cost != -1:
            dm.acc_module_cost(init_time_cost)
            if not local_args.ifTry:
                dm.save_log('module_cost', 'a', 'init_cost:' + s2hms(init_time_cost))

    for epoch in range(local_args.start_epoch, local_args.run_p.pretrain_epoch + 1):

        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])

        loss, batch_time = pretrain(model, pretrain_loader, optimizer, criterion, local_args, epoch)

        if epoch >= local_args.center.warmup_epochs and epoch != local_args.run_p.pretrain_epoch and (
                epoch - local_args.center.warmup_epochs) % local_args.center.u_interval == 0:
            ts = time.time()
            target_Centers = target_Center(eval_loader, model.module.encoder_q, len_ds,
                                           True if local_args.local_rank == 0 else False).cpu()
            assert len(target_Centers) == len_ds
            all_centers = update_Center(target_Centers, pretrain_set.centers, local_args.center.o_ratio)
            assert len(all_centers) == len_ds
            pretrain_set.centers = all_centers.cpu()
            dm.acc_module_cost(time.time() - ts)

        if local_args.local_rank == 0:
            if not local_args.ifTry:
                dm.save_ckp(model, optimizer, 'byol_state', ['centers', pretrain_set.centers])
            dm.end_pretrain_epoch(model.module, loss, batch_time)

def pretrain_worker_Fcrop_sstd(rank, world_size, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.pretrain_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    pretrain_set, eval_set, init_set = build_Fcrop_datasets(local_args.dataset)
    len_ds = len(pretrain_set)

    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set, shuffle=True)
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=pretrain_sampler,
        drop_last=True
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=eval_sampler,
        drop_last=False
    )

    init_loader = torch.utils.data.DataLoader(
        init_set,
        batch_size=1,
        drop_last=False
    )

    local_args.byol.encoder_q = build_model(local_args.model)
    local_args.byol.encoder_k = build_model(local_args.model)
    model = build_model(local_args.byol)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank],
                                                      find_unused_parameters=True)

    criterion = build_criterion(local_args.loss.pretrain).cpu()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    if local_args.args.resume is not None:
        checkpoint = torch.load(os.path.join('./', local_args.args.resume, 'ckp', 'ckp.pth'), map_location='cuda')
        model.load_state_dict(checkpoint['byol_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        local_args.start_epoch = checkpoint['epoch'] + 1
        print('start_epoch = ' + str(local_args.start_epoch))
        pretrain_set.centers = checkpoint['centers'].cpu()

        init_time_cost = -1
        local_args.data = checkpoint['data_manager_param']['data']
        local_args.run_stime = checkpoint['data_manager_param']['run_stime']
        local_args.module_cost = checkpoint['data_manager_param']['module_cost']
        local_args.best_acc1 = checkpoint['data_manager_param']['best_acc1']
        local_args.best_acc5 = checkpoint['data_manager_param']['best_acc5']
        local_args.best_acc1_epoch = checkpoint['data_manager_param']['best_acc1_epoch']
        local_args.best_acc5_epoch = checkpoint['data_manager_param']['best_acc5_epoch']

    else:
        if os.path.exists(os.path.join('./', local_args.dataset.root, 'center_ckp.pth')):
            checkpoint = torch.load(os.path.join('./', local_args.dataset.root, 'center_ckp.pth'), map_location='cuda')
            all_centers = checkpoint['init_centers']
            init_time_cost = checkpoint['init_time_cost']
        else:
            all_centers, init_time_cost = get_Center(init_loader)
            save_centers(local_args.dataset.root, all_centers, init_time_cost)
        assert len(all_centers) == len_ds
        print("Init Time Consumption: " + s2hms(init_time_cost))
        pretrain_set.centers = all_centers.cpu()
        local_args.start_epoch = 1
        local_args.data = []
        local_args.run_stime = time.time()
        local_args.module_cost = 0
        local_args.best_acc1 = 0
        local_args.best_acc5 = 0
        local_args.best_acc1_epoch = 0
        local_args.best_acc5_epoch = 0

    cudnn.benchmark = True

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
        if init_time_cost != -1:
            dm.acc_module_cost(init_time_cost)
            if not local_args.ifTry:
                dm.save_log('module_cost', 'a', 'init_cost:' + s2hms(init_time_cost))

    for epoch in range(local_args.start_epoch, local_args.run_p.pretrain_epoch + 1):

        sstd = get_sstd(epochs=local_args.run_p.pretrain_epoch, epoch=epoch, start_sstd=local_args.center.s_std,
                        min_sstd=local_args.center.e_std)
        # sstd = get_sstd_cos(epochs=local_args.run_p.pretrain_epoch, epoch=epoch, start_sstd=local_args.center.s_std, min_sstd=local_args.center.e_std)
        pretrain_set.sstd = sstd

        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])

        loss, batch_time = pretrain(model, pretrain_loader, optimizer, criterion, local_args, epoch)

        if epoch >= local_args.center.warmup_epochs and epoch != local_args.run_p.pretrain_epoch and (
                epoch - local_args.center.warmup_epochs) % local_args.center.u_interval == 0:
            ts = time.time()
            target_Centers = target_Center(eval_loader, model.module.encoder_q, len_ds,
                                           True if local_args.local_rank == 0 else False).cpu()
            assert len(target_Centers) == len_ds
            all_centers = update_Center(target_Centers, pretrain_set.centers, local_args.center.o_ratio)
            assert len(all_centers) == len_ds
            pretrain_set.centers = all_centers.cpu()
            dm.acc_module_cost(time.time() - ts)

        if local_args.local_rank == 0:
            if not local_args.ifTry:
                dm.save_ckp(model, optimizer, 'byol_state', ['centers', pretrain_set.centers])
            dm.end_pretrain_epoch(model.module, loss, batch_time)
