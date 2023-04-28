import sys
sys.path.append('./model')
sys.path.append('./util')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import VG_generator
from DINet import DINet
from transalnet_customized import TranSalNet
from salicon_model import SALICON
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os
import time
import gc
import json

parser = argparse.ArgumentParser(description='Prototype dissection with Visual Genome')
parser.add_argument('--img_dir', type=str, default=None, help='Directory to image files')
parser.add_argument('--sg_dir', type=str, default=None, help='Directory to scene graph annotation')
parser.add_argument('--weights', type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--batch_size', type=int, default=32, help='Defining batch size for training (default: 150)')
parser.add_argument('--num_proto', type=int, default=512, help='Number of prototype')
parser.add_argument('--model', type=str, default='dinet', help='model to be analyzed')
parser.add_argument('--threshold', type=float, default=0.8, help='threshold for map binarization')
parser.add_argument('--adaptive', type=bool, default=True, help='using adaptive threshold or not')

args = parser.parse_args()

def compute_iou(proto_map, concept_seg, adaptive_threshold=None):
    """ Compute the Intersection over Union (IoU)
        score between two maps.
    """
    # binarize the proto map with a threshold
    if adaptive_threshold is None:
        proto_map[proto_map>args.threshold] = 1
    else:
        proto_map[proto_map>adaptive_threshold] = 1

    proto_map[proto_map<1] = 0
    return np.logical_and(proto_map, concept_seg).sum()/np.logical_or(proto_map, concept_seg).sum()

def prototype_dissection():
    """
    Compute the alignment between prototype activation map and concept
    segmentation in the Visual Genome dataset.
    """
    probe_data = VG_generator(args.img_dir, args.sg_dir)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=4)

    # load fully trained model (partial weights)
    if args.model == 'dinet':
        model = DINet(256, True, args.num_proto, False)
    elif args.model == 'salicon':
        model = SALICON(True, args.num_proto, False)
    elif args.model == 'transalnet':
        model = TranSalNet(True, args.num_proto, False).cuda()
    else:
        assert 0, 'model not supported for analysis yet'


    model.load_state_dict(torch.load(args.weights), strict=False)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    # compute adaptive threshold for each prototype
    if args.adaptive:
        proto_distribution = json.load(open(
                                        'proto_distribution_sal_'+args.model+'.json'))
        adaptive_threshold = dict()
        for proto_idx in proto_distribution:
            accumulated_prob = 0
            tmp_threshold = 0
            for idx, threshold in enumerate(proto_distribution[proto_idx]):
                accumulated_prob += proto_distribution[proto_idx][threshold]
                if accumulated_prob>=args.threshold or idx==len(proto_distribution[proto_idx])-2:
                    tmp_threshold = accumulated_prob
                    break
            adaptive_threshold[int(proto_idx)] = tmp_threshold

    # alignment score between
    proto2concept = [{} for _ in range(args.num_proto)]

    start = time.time()
    with torch.no_grad():
        for batch_idx,(img, img_id) in enumerate(probe_loader):
            # generate the prototype heatmaps
            img = Variable(img).cuda()
            # proto_sim = model(img)
            proto_sim = model(img, probe=True) # for saliency model
            proto_sim = proto_sim.data.cpu().numpy()

            # compare with concept segmentation for each sample
            img_id = img_id[0]
            for i in range(len(proto_sim)):
                concept_seg = probe_data.decode_vg(img_id[i])

                for concept in concept_seg:
                    # exclude invalid ground truth
                    if concept_seg[concept].sum()==0:
                        continue

                    # iterate through all prototypes
                    for proto_idx in range(args.num_proto):
                        proto_map = proto_sim[i, proto_idx, :, :] # for saliency model

                        if not args.adaptive:
                            align_score = compute_iou(proto_map, concept_seg[concept])
                        else:
                            align_score = compute_iou(proto_map, concept_seg[concept], adaptive_threshold[proto_idx])

                        if concept not in proto2concept[proto_idx]:
                            proto2concept[proto_idx][concept] = [align_score]
                        else:
                            proto2concept[proto_idx][concept].append(align_score)

            if (batch_idx+1)%20 == 0:
                print('Finished %d samples, time spent: %.3f' %(
                            (batch_idx+1)*args.batch_size, time.time()-start))

    # normalize and re-rank the alignment scores
    for i in range(args.num_proto):
        for concept in proto2concept[i]:
            proto2concept[i][concept] = np.mean(proto2concept[i][concept])
        proto2concept[i] = {k : v for k, v in sorted(
                                                proto2concept[i].items(), key=lambda item:item[1], reverse=True)}

    with open('proto_dessection_result_'+args.model+'.json', 'w') as f:
        json.dump(proto2concept, f)

prototype_dissection()
