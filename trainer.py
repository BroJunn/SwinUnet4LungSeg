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
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler
from utils import masked_image, tensor2array

def trainer_tod(args, model, snapshot_path):
    from datasets.dataset_tod import Tod_dataset
    logging_path = os.path.join(snapshot_path, datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Tod_dataset(path_dir=os.path.join(args.root_path, 'train'))
    print("The length of train set is: {}".format(len(db_train)))
    db_val= Tod_dataset(path_dir=os.path.join(args.root_path, 'val'))
    print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.StepLR_param[0], gamma=args.StepLR_param[1])
    
    writer = SummaryWriter(logging_path)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 1e10
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        model.train()
        loss_acc_train = 0.0
        for _, sampled_batch in tqdm(enumerate(trainloader)):
            input, gt = sampled_batch['input'], sampled_batch['gt']
            input = input.unsqueeze(1)
            input, gt = input.to(torch.float32).cuda(), gt.cuda()
            outputs = model(input)
            outputs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            loss_acc_train += loss
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss_ce_train', loss, iter_num)

            # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % args.interval_vis_train == 0:
                vis_output = masked_image(tensor2array(input[0][0]), tensor2array(outputs[0]))
                vis_output = vis_output.transpose(2, 0, 1) / 255.0
                writer.add_image('train/Image_output', vis_output, iter_num)
                vis_gt = masked_image(tensor2array(input[0][0]), tensor2array(gt[0]))
                vis_gt = vis_gt.transpose(2, 0, 1) / 255.0
                # import cv2
                # cv2.imwrite('1.png', vis_gt.transpose(1, 2, 0)*255)
                writer.add_image('train/Image_gt', vis_gt, iter_num)
        scheduler.step()

        writer.add_scalar('info/avg_loss_ce_train', loss_acc_train / len(db_train) / batch_size, epoch_num)
        logging.info('epoch_num %d : avg_loss_ce_train : %f' % (epoch_num, loss_acc_train / len(db_train) / batch_size))


        model.eval()
        loss_acc_val = 0.0
        with torch.no_grad():
            for j_batch, val_batch in tqdm(enumerate(valloader)):
                input, gt = val_batch['input'], val_batch['gt']
                input = input.unsqueeze(1)
                input, gt = input.to(torch.float32).cuda(), gt.cuda()
                outputs = model(input)
                outputs = torch.softmax(outputs, dim=1)
                loss_val = criterion(outputs, gt)
                loss_acc_val += loss_val

                if j_batch % args.interval_vis_val == 0:
                    vis_output_val = masked_image(tensor2array(input[0][0]), tensor2array(outputs[0]))
                    vis_output_val = vis_output_val.transpose(2, 0, 1) / 255.0
                    writer.add_image('val/Image_output', vis_output_val, epoch_num)
                    vis_gt_val = masked_image(tensor2array(input[0][0]), tensor2array(gt[0]))
                    vis_gt_val = vis_gt_val.transpose(2, 0, 1) / 255.0
                    writer.add_image('val/Image_gt', vis_gt_val, epoch_num)

        avg_samp_loss = loss_acc_val / len(db_val)
        writer.add_scalar('info/avg_loss_ce_val', avg_samp_loss, epoch_num)
        logging.info('epoch_num %d : avg_loss_ce_val : %f' % (epoch_num, avg_samp_loss))

        if avg_samp_loss < best_performance:
            best_performance = avg_samp_loss
            save_mode_path = os.path.join(logging_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
        if epoch_num >= max_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"