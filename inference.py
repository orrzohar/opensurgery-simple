from tracker.byte_tracker import BYTETracker

from retinanet.inference_dataloader import InferenceDataset, collater, UnNormalizer
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms

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
    parser = argparse.ArgumentParser(description='Surgery Hand and Keypoint Detection on Video', add_help=False)
    ## general
    parser.add_argument('--video_root', help='Dataset type, must be one of csv or coco.', type=str, default="video")
    parser.add_argument('--video', help='Dataset type, must be one of csv or coco.', type=str, default="output.mp4")
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
    return parser


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    
def main(parser):
    use_gpu = parser.use_gpu and torch.cuda.is_available()
    st = time.time()
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
                                     parser.video,
                                     parser.video_clip_length,
                                     parser.fps,
                                     num_classes=4,
                                     min_side=parser.min_side, 
                                     max_side=parser.max_side,
                                     csv_classes=parser.csv_classes)


    
    if parser.viz:
        out_video = parser.video_root+"/" + parser.video[:-4] + "_detections.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_tracked = cv2.VideoWriter(out_video, fourcc, parser.fps, (video_dataset.width, video_dataset.height))
        
    video_sampler = SequentialSampler(video_dataset)
    
    video_dataset = DataLoader(video_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=parser.num_workers,
                               sampler=video_sampler)
    
    output_json={}
    frame_id=0
    print('Elapsed time: {}'.format(time.time() - st)) 
    for idx, data in enumerate(video_dataset):
        with torch.no_grad():
            print('Elapsed time: {}'.format(time.time() - st)) 
            st = time.time()
            if use_gpu:
                scores, classification, transformed_anchors = retinanet(data[0].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data[0].float())

            print('Elapsed time: {}'.format(time.time() - st)) 
            st = time.time()
            idxs = [np.where(s.cpu() > 0.5) for s in scores]
            
            for i in range(len(scores)):
                dets = torch.cat((transformed_anchors[i], classification[i].unsqueeze(1)), dim=1)[idxs[i][0],:]
                output_json[frame_id] = dets.detach().cpu().numpy()
                frame_id+=1
                                
            jj=0
            if parser.viz:
                for i in range(len(classification)):
                    img = np.array(255 * unnormalize(data[0, i, :, :, :])).copy()
                    img[img < 0] = 0
                    img[img > 255] = 255
                    
                    img = np.transpose(img, (1, 2, 0))
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    for j in range(len(idxs[i][0])):
                        bbox = transformed_anchors[i][idxs[i][0][j], :]
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2])
                        y2 = int(bbox[3])
                        label_name = video_dataset.dataset.labels[int(classification[i][idxs[i][0][j]])]
                        
                        b = np.array((x1, y1, x2, y2)).astype(int)
                        cv2.putText(img, label_name, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        cv2.putText(img, label_name, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                        
                    video_tracked.write(np.uint8(img))
                           
    if parser.viz:
        video_tracked.release()
    tmp = np.concatenate([np.column_stack((arr, np.full((arr.shape[0], 1), key))) for key, arr in output_json.items()])
    tmp = pd.DataFrame(tmp, columns = ['x1','y1','x2','y2','l','frame'])
    tmp.to_csv(f'{parser.video_root}/{parser.video[:-4]}_detection_results.csv')
    return


if __name__ == '__main__':
    t1=time.time()
    parser = argparse.ArgumentParser('Surgery Hand tracking and metric calculations', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    print(f'runtime: {time.time()-t1} [s]')