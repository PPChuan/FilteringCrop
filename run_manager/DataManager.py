import os.path
import os

import torch

import time
import json
import pandas as pd
from IPython import display
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_log(path, filename: str, log: str):
    check_dir(path)
    with open(os.path.join(path, filename + '.txt'), 'w', encoding='utf-8') as file:
        file.write(log)


def save_data(path, filename: str, data):
    check_dir(path)
    pd.DataFrame(data).to_csv(os.path.join(path, filename) + '.csv')
    with open(os.path.join(path, filename) + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def s2hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    d = ''
    if h == 0:
        if m == 0:
            d = str(s) + 's'
        if m != 0:
            d = str(m) + 'm ' + str(s) + 's'
    else:
        d = str(h) + 'h ' + str(m) + 'm ' + str(s) + 's'
    return d


def get_grad_tb(net, count, tb):
    for name, param in net.named_parameters():
        if param.grad != None:
            tb.add_histogram(name, param, count)
            tb.add_histogram(f'{name}.grad', param.grad, count)


class DataManager:
    def __init__(self, cfg):
        t = time.time()

        self.cfg = cfg
        self.ifTry = cfg.ifTry
        self.run_name = cfg.run_name
        self.log_dir = os.path.abspath(os.path.join('./', self.run_name))
        self.pretrain_log_dir = os.path.join(self.log_dir, 'pretrain')
        self.ft_log_dir = os.path.join(self.log_dir, 'ft')
        self.tb_dir = os.path.abspath(os.path.join('./runs', self.run_name))

        self.run_stime = t

        self.epoch_stime = 0
        self.epoch_count = 0
        self.epoch_lr = 0

        self.data = []

        self.Com_tb = None
        self.Pretrain_tb = None
        self.Ft_tb = None
        self.Val_tb = None

        self.best_acc1 = 0
        self.best_acc5 = 0
        self.best_acc1_epoch = 0
        self.best_acc5_epoch = 0

        if not self.ifTry:
            save_log(self.log_dir, 'run_params', str(vars(cfg)))

    def reset(self, pt: str):
        if self.epoch_count == 0:
            self.data = []
            self.epoch_stime = 0
            self.epoch_count = 0
            self.epoch_lr = 0

            if pt == 'pretrain':
                if self.Pretrain_tb is None:
                    self.Pretrain_tb = SummaryWriter(log_dir=os.path.join(self.tb_dir, 'PreTrain'))
            elif pt == 'ft':
                if self.Com_tb is None:
                    self.Com_tb = SummaryWriter(log_dir=os.path.join(self.tb_dir, 'Compare'))
                if self.Ft_tb is None:
                    self.Ft_tb = SummaryWriter(log_dir=os.path.join(self.tb_dir, 'FineTune'))
                if self.Val_tb is None:
                    self.Val_tb = SummaryWriter(log_dir=os.path.join(self.tb_dir, 'Val'))

    def begin_pretrain_epoch(self, lr):
        self.reset('pretrain')
        t = time.time()
        self.epoch_lr = lr
        self.epoch_stime = t
        self.epoch_count += 1

    def end_pretrain_epoch(self, model, loss, batch_time):
        t = time.time()
        run_duration = t - self.run_stime
        epoch_duration = t - self.epoch_stime

        pretrain_speed = round(self.cfg.run_p.pretrain_batch_size / batch_time, 2)
        time_remaining = (self.cfg.run_p.pretrain_epoch - self.epoch_count) * epoch_duration

        pretrain_results = OrderedDict()
        pretrain_results['model(Pretrain)'] = self.cfg.run_p.network
        for k, v in vars(self.cfg.run_p).items():
            if k == 'pretrain_epoch':
                pretrain_results[k] = str(self.epoch_count) + '/' + str(v)
                pretrain_results['loss'] = loss
                pretrain_results['lr'] = self.epoch_lr
                pretrain_results['time remaining'] = s2hms(time_remaining)
                pretrain_results['pretrain speed(image/s)'] = pretrain_speed
                pretrain_results['epoch duration'] = s2hms(epoch_duration)
                pretrain_results['run duration'] = s2hms(run_duration)

        self.data.append(pretrain_results)
        pretrain_df = pd.DataFrame([pretrain_results])
        pretrain_df = pretrain_df.round({'loss': 4, 'lr': 4, 'pretrain speed(image/s)': 1})
        display.display(pretrain_df)

        if not self.ifTry:
            self.Pretrain_tb.add_scalar('Pretrain_Loss', loss, self.epoch_count)
            get_grad_tb(model, self.epoch_count, self.Pretrain_tb)
            save_data(self.pretrain_log_dir, 'pretrain_log', self.data)
            self.save_pretrain_model(model)
            if self.epoch_count == self.cfg.run_p.pretrain_epoch:
                self.Pretrain_tb.close()
        if self.epoch_count == self.cfg.run_p.pretrain_epoch:
            self.epoch_count = 0

    def begin_ft_epoch(self, lr):
        self.reset('ft')
        t = time.time()
        self.epoch_lr = lr
        self.epoch_stime = t
        self.epoch_count += 1

    def end_ft_epoch(self, model, t_loss, v_loss, t_batch_time, v_batch_time, acc1, acc5):

        t = time.time()

        epoch_duration = t - self.epoch_stime
        run_duration = t - self.run_stime
        time_remaining = (self.cfg.run_p.ft_epoch - self.epoch_count) * epoch_duration

        v_acc = round(acc1 / self.cfg.val_set_len, 4)
        v_acc_t5 = round(acc5 / self.cfg.val_set_len, 4)

        if self.best_acc1 < v_acc:
            self.best_acc1 = v_acc
            self.save_ft_best_model(model)
            self.best_acc1_epoch = self.epoch_count
        if self.best_acc5 < v_acc_t5:
            self.best_acc5 = v_acc_t5
            self.best_acc5_epoch = self.epoch_count

        train_speed = round(self.cfg.run_p.ft_batch_size / t_batch_time, 2)
        pre_speed = round(self.cfg.run_p.val_batch_size / v_batch_time, 2)

        ft_results = OrderedDict()
        ft_results['model(Ft)'] = self.cfg.run_p.network
        for k, v in vars(self.cfg.run_p).items():
            if k == 'ft_epoch':
                ft_results[k] = str(self.epoch_count) + '/' + str(v)
                ft_results['val_acc_top1'] = v_acc
                ft_results['val_acc_top5'] = v_acc_t5
                ft_results['Best_acc1'] = str(self.best_acc1) + '(epoch:' + str(self.best_acc1_epoch) + ')'
                ft_results['Best_acc5'] = str(self.best_acc5) + '(epoch:' + str(self.best_acc5_epoch) + ')'
                ft_results['train_loss'] = t_loss
                ft_results['val_loss'] = v_loss
                ft_results['lr'] = self.epoch_lr
                ft_results['time remaining'] = s2hms(time_remaining)
                ft_results['train_speed(images/s)'] = train_speed
                ft_results['pre_speed(images/s)'] = pre_speed
                ft_results['epoch duration'] = s2hms(epoch_duration)
                ft_results['run duration'] = s2hms(run_duration)

        self.data.append(ft_results)
        ft_df = pd.DataFrame([ft_results])
        ft_df = ft_df.round(
            {'val_acc_top1': 4, 'val_acc_top5': 4, 'train_loss': 4, 'val_loss': 4, 'lr': 4, 'train_speed(images/s)': 1,
             'pre_speed(images/s)': 1})
        display.display(ft_df)

        if not self.ifTry:
            self.Val_tb.add_scalars('Val_Accuracy',
                                    {'Val_Acc_Top1': v_acc,
                                     'Val_Acc_Top5': v_acc_t5}, self.epoch_count)
            self.Val_tb.add_scalar('Val_Loss', v_loss, self.epoch_count)

            self.Ft_tb.add_scalar('Train_Loss', t_loss, self.epoch_count)

            self.Com_tb.add_scalars('Compare_Loss',
                                    {'Train_Loss': t_loss,
                                     'Val_Loss': v_loss}, self.epoch_count)
            self.Com_tb.add_scalars('Compare_Val',
                                    {'Val_Acc': v_acc,
                                     'Val_Loss': v_loss}, self.epoch_count)
            get_grad_tb(model, self.epoch_count, self.Ft_tb)
            save_data(self.ft_log_dir, 'ft_log', self.data)
            self.save_ft_model(model)
            if self.epoch_count == self.cfg.run_p.ft_epoch:
                self.Val_tb.close()
                self.Ft_tb.close()
                self.Com_tb.close()
        if self.epoch_count == self.cfg.run_p.ft_epoch:
            self.epoch_count = 0

    def save_pretrain_model(self, model):
        if not self.ifTry:
            path = os.path.join(self.pretrain_log_dir, 'checkpoint')
            check_dir(path)
            if self.epoch_count == self.cfg.run_p.pretrain_epoch:
                torch.save(model, os.path.join(path, 'last_model.pth'))
                torch.save(model.state_dict(), os.path.join(path, 'last_weights.pth'))
                print('===> Finish Save Final Model')
            elif isinstance(self.cfg.save_interval.pretrain, int):
                if self.epoch_count % self.cfg.save_interval.pretrain == 0:
                    torch.save(model.state_dict(), os.path.join(path, 'weights_epoch' + str(self.epoch_count) + '.pth'))
                    print('===> Finish Save Process')
            elif isinstance(self.cfg.save_interval.pretrain, list):
                if self.epoch_count in self.cfg.save_interval.pretrain:
                    torch.save(model.state_dict(), os.path.join(path, 'weights_epoch' + str(self.epoch_count) + '.pth'))
                    print('===> Finish Save Process')
        else:
            print('===> Just Try! Continue...')

    def save_ft_model(self, model):
        if not self.ifTry:
            path = os.path.join(self.ft_log_dir, 'checkpoint')
            check_dir(path)
            if isinstance(self.cfg.save_interval.pretrain, int):
                if self.epoch_count % self.cfg.save_interval.pretrain == 0:
                    torch.save(model.state_dict(), os.path.join(path, 'weights_epoch' + str(self.epoch_count) + '.pth'))
                    print('===> Finish Save Process')
            elif isinstance(self.cfg.save_interval.pretrain, list):
                if self.epoch_count in self.cfg.save_interval.pretrain:
                    torch.save(model.state_dict(), os.path.join(path, 'weights_epoch' + str(self.epoch_count) + '.pth'))
                    print('===> Finish Save Process')
        else:
            print('===> Just Try! Continue...')

    def save_ft_best_model(self, model):
        if not self.ifTry:
            path = os.path.join(self.ft_log_dir, 'checkpoint')
            check_dir(path)
            torch.save(model, os.path.join(path, 'best_model.pth'))
            torch.save(model.state_dict(), os.path.join(path, 'best_weights.pth'))
            print('===> Finish Save Best Model')
        else:
            print('===> Just Try! Continue...')
