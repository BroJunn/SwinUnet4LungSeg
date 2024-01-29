import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet
from tester import tester_tod
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/root/autodl-tmp/data', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='covid-19', help='experiment_name')
parser.add_argument('--output_dir', type=str, default='./logger', help='output dir')  \

parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str,
                    default='configs/swin_tiny_patch4_window8_512_lite.yaml',
                    metavar="FILE", help='path to config file', )
parser.add_argument('--interval_vis_test', type=int,
                    default=5, help='how many iterations to save visualization')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'covid-19': {
            'root_path': args.root_path,
            'num_classes': 3
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_state_dict(torch.load(config.MODEL.PRETRAIN_CKPT, map_location="cuda:0"))

    tester = {'covid-19': tester_tod}
    tester[dataset_name](args, net, args.output_dir)