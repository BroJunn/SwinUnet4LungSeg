import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.optim import lr_scheduler

def trainer_tod(args, model, snapshot_path):
    from datasets.dataset_tod import Tod_dataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
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
    # model.train()
    mse_loss = MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 1e10
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # start = datetime.now()
        model.train()
        loss_acc_train = 0.0
        for i_batch, sampled_batch in tqdm(enumerate(trainloader)):
            input, gt = sampled_batch['input'], sampled_batch['gt']
            input, gt = input.unsqueeze(1), gt.unsqueeze(1)
            input, gt = input.cuda(), gt.cuda()
            outputs = model(input)
            loss = mse_loss(outputs, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            loss_acc_train += loss
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss_mse_train', loss, iter_num)

            # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 200 == 0:
                image_input = input[0, 0:1, :, :]
                image_input = (image_input - image_input.min()) / (image_input.max() - image_input.min())
                writer.add_image('train/Image_input', image_input, iter_num)
                image_output = outputs[0, 0:1, :, :]
                image_output = (image_output - image_output.min()) / (image_output.max() - image_output.min())
                writer.add_image('train/Image_output', image_output, iter_num)
                image_gt = gt[0, 0:1, :, :]
                image_gt = (image_gt - image_gt.min()) / (image_gt.max() - image_gt.min())
                writer.add_image('train/Image_gt', image_gt, iter_num)
        scheduler.step()

        # end = datetime.now()
        # timeDiff = end - start
        # logging.info('Time for one epoch: %f min.' % (timeDiff.total_seconds() / 60))

        writer.add_scalar('info/loss_mse_train', loss_acc_train / len(db_train) / batch_size, epoch_num)
        logging.info('epoch_num %d : loss_mse_train : %f' % (epoch_num, loss_acc_train / len(db_train) / batch_size))

        model.eval()
        loss_acc_val = 0.0
        with torch.no_grad():
            for j_batch, val_batch in tqdm(enumerate(valloader)):
                input, gt = val_batch['input'], val_batch['gt']
                input, gt = input.unsqueeze(1), gt.unsqueeze(1)
                input, gt = input.cuda(), gt.cuda()
                outputs = model(input)
                loss_val = mse_loss(outputs, gt)
                loss_acc_val += loss_val

                if j_batch in [0, 20, 40, 60, 100]:
                    image_input = input[0, 0:1, :, :]
                    image_input = (image_input - image_input.min()) / (image_input.max() - image_input.min())
                    writer.add_image('val/Image_input', image_input, epoch_num)
                    image_output = outputs[0, 0:1, :, :]
                    image_output = (image_output - image_output.min()) / (image_output.max() - image_output.min())
                    writer.add_image('val/Image_output', image_output, epoch_num)
                    image_gt = gt[0, 0:1, :, :]
                    image_gt = (image_gt - image_gt.min()) / (image_gt.max() - image_gt.min())
                    writer.add_image('val/Image_gt', image_gt, epoch_num)

        avg_samp_loss = loss_acc_val / len(db_val)
        writer.add_scalar('info/loss_mse_val', avg_samp_loss, epoch_num)
        logging.info('epoch_num %d : loss_mse_val : %f' % (epoch_num, avg_samp_loss))

        if avg_samp_loss < best_performance:
            best_performance = avg_samp_loss
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
        if epoch_num >= max_epoch - 1:
            iterator.close()
            break
        # save_interval = 10  # int(max_epoch/6)
        # if (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    writer.close()
    return "Training Finished!"