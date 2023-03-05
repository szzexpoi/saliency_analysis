import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import os
import time
import gc
import tensorflow as tf
from dataloader import salicon
from evaluation import cal_cc_score, cal_sim_score, cal_kld_score, cal_auc_score, cal_nss_score, add_center_bias
from DINet import DINet
from transalnet_customized import TranSalNet
from salicon_model import SALICON
from loss import NSS, CC, KLD, cross_entropy
import cv2
import json
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Saliency prediction on SALICON')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--img_dir', type=str, default=None, help='Directory to the image data')
parser.add_argument('--fix_dir', type=str, default=None, help='Directory to the raw fixation file')
parser.add_argument('--anno_dir', type=str, default=None, help='Directory to the saliency maps')
parser.add_argument('--width', type=int, default=640, help='Width of input data')
parser.add_argument('--height', type=int, default=480, help='Height of input data')
parser.add_argument('--clip', type=float, default=-1, help='Gradient clipping')
parser.add_argument('--batch', type=int, default=10, help='Batch size')
parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay factor')
parser.add_argument('--lr_decay_step', type=int, default=2, help='Learning rate decay step')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
parser.add_argument('--center_bias', type=bool, default=True, help='Adding center bias or not')
parser.add_argument('--feat_dim', type=int, default=512, help='Feature dimension before the last layer')
parser.add_argument('--use_proto', type=bool, default=False, help='using fractoization or not')
parser.add_argument('--num_proto', type=int, default=512, help='number of prototypes for factorization')
parser.add_argument('--weights', type=str, default=None, help='Weights to be loaded')
parser.add_argument('--model', type=str, default=None, help='selection of saliency model')
parser.add_argument('--second_phase', type=bool, default=False, help='Second phase training or not?')
parser.add_argument('--use_interaction', type=bool, default=False, help='For analyzing interaction or not?')

args = parser.parse_args()


transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def add_summary_value(writer, key, value, iteration): #tensorboard visualization
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, epoch):
	"adatively adjust lr based on iteration"
	if epoch >= 1: #30-adam
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * (args.lr_decay ** (epoch/args.lr_decay_step))

def training():
    """ Main function for training different saliency models
    """
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint)
    train_data = salicon(args.anno_dir, args.fix_dir, args.img_dir, args.width,
                        args.height, 'train', transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch,
                        shuffle=True, num_workers=8)

    test_data = salicon(args.anno_dir, args.fix_dir, args.img_dir, args.width,
                    args.height, 'val', transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch,
                    shuffle=False, num_workers=8)

    # model construction
    if args.model == 'dinet':
        model = DINet(args.feat_dim, args.use_proto, args.num_proto,
                    args.second_phase, args.use_interaction)
    elif args.model == 'salicon':
        model = SALICON(args.use_proto, args.num_proto,
                    args.second_phase, args.use_interaction)
    elif args.model == 'transalnet':
        model = TranSalNet(args.use_proto, args.num_proto,
                        args.second_phase, args.use_interaction)
    else:
        assert 0, "model not yet supported"

    # for fine-tuning (second phase), load pretrained model
    if args.second_phase:
        model.load_state_dict(torch.load(args.weights), strict=False)

    model = nn.DataParallel(model).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=1e-7) #1e-8

    def train(iteration):
        """ Training for a single epoch
        """
        model.train()
        avg_loss = 0
        for i, (img, sal_map, fix, img_id) in enumerate(trainloader):
            img, sal_map, fix = img.cuda(), sal_map.cuda(), fix.cuda()
            optimizer.zero_grad()
            pred = model(img)
            loss = NSS(pred, fix) + KLD(pred, sal_map) + CC(pred, sal_map)
            loss.backward()
            if args.clip != -1 :
                clip_gradient(optimizer, args.clip) #gradient clipping without normalization
            optimizer.step()

            avg_loss = (avg_loss*np.maximum(0,i) + loss.data.cpu().numpy())/(i+1)
            if i%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('training loss', avg_loss,step=iteration)
            iteration += 1
        return iteration

    def test(iteration):
        """ Validation
        """
        model.eval()
        nss_score = []
        cc_score = []
        auc_score = []
        sim_score = []
        kld_score = []
        for i, (img, sal_map, fix, img_id) in enumerate(testloader):
            img = img.cuda()
            pred = model(img)
            pred = pred.data.cpu().numpy()
            sal_map = sal_map.data.numpy()
            fix = fix.data.numpy()

            # computing score for each data
            for j in range(len(img)):
                cur_pred = pred[j].squeeze()

                if args.center_bias:
                    cur_pred = add_center_bias(cur_pred)
                cc_score.append(cal_cc_score(cur_pred, sal_map[j]))
                sim_score.append(cal_sim_score(cur_pred, sal_map[j]))
                kld_score.append(cal_kld_score(cur_pred, sal_map[j]))
                nss_score.append(cal_nss_score(cur_pred, fix[j]))
                auc_score.append(cal_auc_score(cur_pred, fix[j]))

        with tf_summary_writer.as_default():
            tf.summary.scalar('NSS', np.mean(nss_score), step=iteration)
            tf.summary.scalar('CC', np.mean(cc_score), step=iteration)
            tf.summary.scalar('AUC', np.mean(auc_score), step=iteration)
            tf.summary.scalar('SIM', np.mean(sim_score), step=iteration)
            tf.summary.scalar('KLD', np.mean(kld_score), step=iteration)


        return np.mean(cc_score)

    iteration = 0
    best_score = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch+1)
        iteration = train(iteration)
        cur_score = test(iteration)
        torch.save(model.module.state_dict(), os.path.join(args.checkpoint,'model.pth'))
        if cur_score > best_score:
            best_score = cur_score
            torch.save(model.module.state_dict(), os.path.join(args.checkpoint,'model_best.pth'))

def compute_threshold():
    """ Compute the adaptive threshold for prototype dissection
    """
    test_data = salicon(args.anno_dir, args.fix_dir, args.img_dir, args.width,
                    args.height, 'val', transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch,
                    shuffle=False, num_workers=8)

    # loading the model
    if args.model == 'dinet':
        model = DINet(args.feat_dim, args.use_proto, args.num_proto, False, False)
    elif args.model == 'salicon':
        model = SALICON(args.use_proto, args.num_proto, False, False)
    elif args.model == 'transalnet':
        model = TranSalNet(args.use_proto, args.num_proto, False, False)
    else:
        assert 0, "model not yet supported"

    model.load_state_dict(torch.load(args.weights), strict=True)
    model = nn.DataParallel(model).cuda()
    model.eval()

    # compute the distribution of prototype activation
    proto_freq = dict()
    for proto_idx in range(args.num_proto):
        proto_freq[proto_idx] = dict()
        for interval in np.linspace(0.01, 1, 100):
            interval = round(float(interval), 2)
            proto_freq[proto_idx][interval] = 0
    total = 0

    with torch.no_grad():
        # first compute the proto-specific activation for each image
        for i, (img, sal_map, fix, img_id) in enumerate(testloader):
            img = img.cuda()
            proto_sim = model(img, probe=True) # batch x proto x h x w

            # record the distribution data
            for proto_idx in range(args.num_proto):
                prev = 0
                for interval in np.linspace(0.01, 1, 100):
                    interval = round(float(interval), 2)
                    cur_count = (proto_sim[:, proto_idx]<=interval).sum().data.cpu().numpy()
                    proto_freq[proto_idx][interval] += int(cur_count)-prev
                    prev = int(cur_count)
            total += proto_sim.shape[0]*proto_sim.shape[1]

    # save the distribution of prototype activations
    for proto_idx in range(args.num_proto):
        for interval in np.linspace(0.01, 1, 100):
            interval = round(float(interval), 2)
            proto_freq[proto_idx][interval] /= total

    with open('./proto_distribution_sal_'+args.model+'.json', 'w') as f:
        json.dump(proto_freq, f)


def interaction_analysis():
    """ Analyzing the interaction with
        prototype matching and global self-attention.
    """
    test_data = salicon(args.anno_dir, args.fix_dir, args.img_dir, args.width,
                    args.height, 'val', transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch,
                    shuffle=False, num_workers=8)

    # loading the model (remember to change the settings of model)
    model = DINet(args.feat_dim, True, args.num_proto, True, True)
    model.load_state_dict(torch.load(args.weights), strict=True)
    model = nn.DataParallel(model).cuda()
    model.eval()

    # computing the average attention pattern
    avg_att = torch.zeros(1200, 1200)
    total = 0

    # recording the dependencies between prototypes
    proto_att = torch.zeros(args.num_proto, args.num_proto)

    # recording evaluation score
    nss_score = []
    cc_score = []
    auc_score = []
    sim_score = []
    kld_score = []

    with torch.no_grad():
        # first compute the proto-specific activation for each image
        for idx, (img, sal_map, fix, img_id) in enumerate(testloader):
            img = img.cuda()
            fix = fix.data.cpu().numpy()
            sal_map = sal_map.data.cpu().numpy()
            proto_sim, att_map = model(img, probe=True) # batch x proto x h x w

            # binarize the prototype matching results (with latter steps)
            proto_sim = proto_sim.data.cpu()
            proto_sim[proto_sim<=args.threshold] = 0
            proto_sim[proto_sim>args.threshold] = 1

            # compute the prototype-wise interactions
            for i in range(len(img)):
                cur_att = att_map[i].data.cpu()
                cur_proto = proto_sim[i]
                tmp_proto_att = torch.mm(
                                torch.mm(cur_proto, cur_att),
                                    cur_proto.transpose(1,0))

                # normalize the interaction
                tmp_proto_att = tmp_proto_att/(tmp_proto_att.sum(-1, keepdim=True)+1e-15)
                proto_att += tmp_proto_att

            # measuring the alignment between aggregated attention and saliency maps
            for i in range(len(img)):
                # filtering the interaction only for salient regions
                cur_att = att_map[i].data.cpu().numpy()
                resize_sal_map = cv2.resize(sal_map[i], (80, 60))
                cur_att = (cur_att*resize_sal_map.reshape([-1, 1])).sum(0)
                cur_att = cur_att.reshape([60, 80])
                cur_att = cv2.resize(cur_att, (640, 480))
                cur_att /= (cur_att.max()+1e-5)

                cur_att = add_center_bias(cur_att)
                cc_score.append(cal_cc_score(cur_att, sal_map[i]))
                sim_score.append(cal_sim_score(cur_att, sal_map[i]))
                kld_score.append(cal_kld_score(cur_att, sal_map[i]))
                nss_score.append(cal_nss_score(cur_att, fix[i]))

    # save the prototype-wise interaction
    proto_att /= (proto_att.max(-1, keepdim=True)[0]+1e-15)
    torch.save(proto_att, 'prototype_interaction.pt')

    # print saliency scores
    print('NSS:', np.mean(nss_score))
    print('CC:', np.mean(cc_score))
    print('KLD:', np.mean(kld_score))
    print('SIM:', np.mean(sim_score))

if args.mode == 'train':
    training()
elif args.mode == 'compute_threshold':
    compute_threshold()
elif args.mode == 'interaction_analysis':
    interaction_analysis()
