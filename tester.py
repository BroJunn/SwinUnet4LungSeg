import argparse
import logging
import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler
from utils import masked_image, tensor2array

def tester_tod(args, model, snapshot_path):
    from datasets.dataset_tod import Tod_dataset
    logging_path = os.path.join(snapshot_path, "TEST_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    db_test= Tod_dataset(path_dir=os.path.join(args.root_path, 'test'))
    print("The length of validation set is: {}".format(len(db_test)))

    valloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    writer = SummaryWriter(logging_path)

    model.eval()
    with torch.no_grad():
        for j_batch, val_batch in tqdm(enumerate(valloader)):
            input, gt = val_batch['input'], val_batch['gt']
            input = input.unsqueeze(1)
            input, gt = input.to(torch.float32).cuda(), gt.cuda()
            outputs = model(input)
            outputs = torch.softmax(outputs, dim=1)

            if j_batch % args.interval_vis_test == 0:
                vis_output_val = masked_image(tensor2array(input[0][0]), tensor2array(outputs[0]))
                vis_output_val = vis_output_val.transpose(2, 0, 1) / 255.0
                writer.add_image('test_output/Image_' + str(j_batch), vis_output_val, j_batch)
                vis_gt_val = masked_image(tensor2array(input[0][0]), tensor2array(gt[0]))
                vis_gt_val = vis_gt_val.transpose(2, 0, 1) / 255.0
                writer.add_image('test_gt/Image_' + str(j_batch), vis_gt_val, j_batch)

    writer.close()
    return "Testing Finished!"