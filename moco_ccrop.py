import torch.multiprocessing as mp

from worker.worker_moco import *
from worker.worker_ft import ft_worker
from run_manager import *
from build.util import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_printoptions(linewidth=150)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--cfg', default=False, type=str, required=True,
                        help='this option is required, for example: simclr/cifar10')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')

    # ifTry=True will not save tensorboard and log
    parser.add_argument('--ifTry', '--i', default='true', type=str,
                        help='True = yes, true, t, y, 1; False = no, false, f, n, 0')
    parser.add_argument('--resume', '--r', default=None, type=str,
                        help='path to resume checkpoint')
    return parser.parse_args()


def main():
    world_size = torch.cuda.device_count()
    print('GPUs on this node:', world_size)
    print('cfg:', args.config)
    cfg.world_size = world_size
    cfg.ifTry = str2bool(args.ifTry)
    cfg.run_name = args.config.replace('/', '_') + time.strftime('_%Y-%m-%d_%H', time.localtime(time.time()))
    cfg.args = args

    print("num_workers = " + str(cfg.run_d.num_workers))
    mp.spawn(pretrain_worker_ccrop, nprocs=world_size, args=(world_size, cfg))

    model_dir = os.path.abspath(os.path.join('./' + cfg.run_name, 'pretrain', 'checkpoint', 'last_weights.pth'))
    cfg.args.f_resume = None
    mp.spawn(ft_worker, nprocs=world_size, args=(world_size, model_dir, cfg))
    print('config:' + args.config)


if __name__ == '__main__':
    args = parser()
    # 设置随机种子
    set_seed(args.seed)
    # 获取配置文件
    cfg = read_cfg(args.config)

    main()
