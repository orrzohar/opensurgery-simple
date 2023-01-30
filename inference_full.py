from tracker.byte_tracker import BYTETracker
from utils import extract_metrics, global_metrics, create_video_with_plot
from retinanet.inference_dataloader import InferenceDataset, collater, UnNormalizer
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from tqdm import tqdm

import numpy as np
import pandas as pd
import argparse
import json
import torch
import os
import time
import cv2

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_args_parser():
    parser = argparse.ArgumentParser(description='Surgery Hand tracking on Video', add_help=False)
    ## general
    parser.add_argument('--video_root', help='Dataset type, must be one of csv or coco.', type=str, default="video")
    parser.add_argument('--vid_name', help='Dataset type, must be one of csv or coco.', type=str, default="output.mp4")
    parser.add_argument('--viz', help='Dataset type, must be one of csv or coco.', type=bool, default=False)
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', default='data/AVOS/detections/hand_tool_class_names.csv')
    parser.add_argument('--video_clip_length', default=20)
    parser.add_argument('--num_workers', default=4)#4

    parser.add_argument('--fps', default=13)
    parser.add_argument('--min_side', default=608)
    parser.add_argument('--max_side', default=1024)
    parser.add_argument('--use_gpu', default=True)

    ## detection model
    parser.add_argument('--model', help='Path to model (.pt) file.', type=str, default="weights/csv_retinanet_98.pt")
    parser.add_argument('--num_classes', help='number of classes', type=int, default=4)

    ## bytracker
    parser.add_argument('--object_tracked', default='hand')

    parser.add_argument('--expn', default=None)
    parser.add_argument('--test', default=True)
    parser.add_argument('--opts', default=None)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--tsize', default=None)
    parser.add_argument('--mot20', default=False)

    parser.add_argument('--conf', default=0.01)
    parser.add_argument('--nms', default=0.2)
    parser.add_argument('--track_thresh', default=0.5)
    parser.add_argument('--track_buffer', default=30)
    parser.add_argument('--match_thresh', default=0.9)
    parser.add_argument('--min_box_area', default=0)
    
    ## metric calculation parameters
    parser.add_argument('--std_sliding_window', default=5)
    
    return parser


def process_tracking(online_targets, args,frame_id, results):
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > args.min_box_area:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            results.append((frame_id, tlwh[:2]+tlwh[2:]/2, tid))
            
    return results


def vis_tracking(online_targets, args,frame_id, results):
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > args.min_box_area:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            results.append((frame_id, tlwh, tid))
            
    return results

def main(parser):
    use_gpu = parser.use_gpu and torch.cuda.is_available()
    retinanet = torch.load(parser.model)
    if parser.use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
            retinanet = torch.nn.DataParallel(retinanet).cuda()

    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()
    unnormalize = UnNormalizer()

    video_dataset = InferenceDataset(parser.video_root,
                                     parser.vid_name,
                                     parser.video_clip_length,
                                     parser.fps,
                                     num_classes=4,
                                     min_side=parser.min_side, 
                                     max_side=parser.max_side,
                                     csv_classes=parser.csv_classes)


    object_tracked = {v: k for k, v in video_dataset.labels.items()}[parser.object_tracked]
    img_size = (video_dataset.height, video_dataset.width)
    
    if parser.viz:
        out_video = parser.video_root+"/" + parser.vid_name[:-4] + "_detections.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_tracked = cv2.VideoWriter(out_video, fourcc, parser.fps, (video_dataset.width, video_dataset.height))
        
    video_sampler = SequentialSampler(video_dataset)
    
    video_dataset = DataLoader(video_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=parser.num_workers,
                               sampler=video_sampler)
    
    output_json={}
    results   = []
    results_viz = []
    frame_id=0
    mot_tracker = BYTETracker(parser)
    for data in tqdm(video_dataset, total=len(video_dataset)) :
        with torch.no_grad():
            if use_gpu:
                scores, classification, transformed_anchors = retinanet(data[0].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data[0].float())

            idxs = [np.where(s.cpu() > 0.5) for s in scores]
            
            for i in range(len(scores)):
                dets = torch.cat((transformed_anchors[i], scores[i].unsqueeze(1), classification[i].unsqueeze(1)), dim=1)[idxs[i][0],:].detach().cpu().numpy()
                output_json[frame_id] = dets
                tracked_dets=dets[dets[:,-1]==object_tracked][:,:-1]
                info_imgs = [img_size[0], img_size[1], frame_id, 0, f'{parser.vid_name[:-4]}_{frame_id}']
                online_targets = mot_tracker.update(tracked_dets, info_imgs, img_size)
                results = process_tracking(online_targets, parser, frame_id, results)
                if parser.viz:
                    results_viz = vis_tracking(online_targets, parser, frame_id, results_viz)
                frame_id+=1

            if parser.viz:
                results_viz_out=pd.DataFrame(results_viz, columns = ['frame_id','coord','tracklet_id'])
                for ii in range(len(scores)):
                    img = np.array(255 * unnormalize(data[0, ii, :, :, :])).copy()
                    img[img < 0] = 0
                    img[img > 255] = 255
                    
                    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
                    img = cv2.cvtColor(img[...,[2,1,0]], cv2.COLOR_BGR2RGB)
                    
                    tracklets = results_viz_out.loc[results_viz_out.frame_id==frame_id-len(scores)+ii]
                    for _, r  in tracklets.iterrows():
                        
                        bbox = r.coord
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2]+bbox[0])
                        y2 = int(bbox[3]+bbox[1])
                        label_name = str(r.tracklet_id)
                        
                        b = np.array((x1, y1, x2, y2)).astype(int)
                        cv2.putText(img, label_name, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        cv2.putText(img, label_name, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                        
                    video_tracked.write(np.uint8(img))
                    
                           
    if parser.viz:
        video_tracked.release()
    tmp = np.concatenate([np.column_stack((arr, np.full((arr.shape[0], 1), key))) for key, arr in output_json.items()])
    tmp = pd.DataFrame(tmp, columns = ['x1','y1','x2','y2','s','l','frame'])
    tmp.to_csv(f'{parser.video_root}/{parser.vid_name[:-4]}_detection_results.csv')
    results = pd.DataFrame(results, columns = ['frame_id','coord','tracklet_id'])
    results.to_csv(f'{parser.video_root}/{parser.vid_name[:-4]}_tracking_results.csv')
    return results
       
    
def run_inference(args):
    t1=time.time()
    
    results = main(args)

    metrics  = extract_metrics(results, args)
       
    g_metrics = global_metrics(metrics, args)
    
    print(f'Total runtime: {time.time()-t1} [s]')
    return metrics, g_metrics

    
if __name__ == '__main__':
    t1=time.time()
    parser = argparse.ArgumentParser('Surgery Hand tracking and metric calculations', parents=[get_args_parser()])
    args = parser.parse_args()
    results = main(args)

    metrics  = extract_metrics(results, args)
       
    g_metrics = global_metrics(metrics, args)
    
    print(f'Total runtime: {time.time()-t1} [s]')
    
