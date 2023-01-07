from retinanet.inference_dataloader import InferenceDataset, collater, UnNormalizer
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pandas as pd
import numpy as np


def moving_std(df, window_size):
    return df.rolling(window_size).std()


def extract_metrics(tracking_outputs, args):
    
    out_x=tracking_outputs.copy()
    out_y=tracking_outputs.copy()
    
    out_x['coord'] = pd.DataFrame(out_x['coord'].values.tolist(), columns=['coord_x', 'coord_y'])['coord_x']
    out_y['coord'] = pd.DataFrame(out_y['coord'].values.tolist(), columns=['coord_x', 'coord_y'])['coord_y']

    out_x = out_x.pivot_table(index = ['frame_id'], values = ['coord'], columns = ['tracklet_id'])
    out_y = out_y.pivot_table(index = ['frame_id'], values = ['coord'], columns = ['tracklet_id'])
    
    vx = out_x.shift()-out_x
    vy = out_y.shift()-out_y
    
    vx_n = vx/np.sqrt(vx ** 2 + vy ** 2)
    vy_n = vy/np.sqrt(vx ** 2 + vy ** 2)
    
    ax = (out_x.shift().shift() + out_x - 2 * out_x.shift())/2
    ay = (out_y.shift().shift() + out_y - 2 * out_y.shift())/2
    
    EoM=np.sqrt(vx**2 + vy**2)
    A1 = ax * vx_n.shift() + ay * vy_n.shift()
    A2 = np.sqrt(ax**2 + ay**2 - A1**2)
    
    mov_std_x  = moving_std(out_x, args.std_sliding_window)
    mov_std_y  = moving_std(out_y, args.std_sliding_window)
    mov_std = np.sqrt(mov_std_x**2+mov_std_y**2)
    
    metrics={
        "EoM": medfilt(EoM.mean(axis=1),3),
        "LinAcc": medfilt(A1.mean(axis=1),3),
        "CentAcc": medfilt(A2.mean(axis=1),3),
        "Jitter": mov_std.mean(axis=1)
    }

    return metrics


def global_metrics(metrics, args):
    glob_met={}
    glob_met["average_velocity"]= np.nanmean(metrics["EoM"])
    glob_met["economy_of_motion"]= np.nansum(metrics["EoM"])

    glob_met["lin_acc_10p"]= np.nanpercentile(metrics["LinAcc"],10)
    glob_met["lin_acc_50p"]= np.nanpercentile(metrics["LinAcc"],50)
    glob_met["lin_acc_90p"]= np.nanpercentile(metrics["LinAcc"],90)

    glob_met["cent_acc_10p"]= np.nanpercentile(metrics["CentAcc"],10)
    glob_met["cent_acc_50p"]=  np.nanpercentile(metrics["CentAcc"],50)
    glob_met["cent_acc_90p"]=  np.nanpercentile(metrics["CentAcc"],90)

    glob_met["jitter_acc_10p"]= np.nanpercentile(metrics["Jitter"],10)
    glob_met["jitter_acc_50p"]=  np.nanpercentile(metrics["Jitter"],50)
    glob_met["jitter_acc_90p"]=  np.nanpercentile(metrics["Jitter"],90)
    out = pd.DataFrame(glob_met, index=[0])
    out.to_csv(args.video_root+args.vid_name[:-4]+"_global_metrics.csv")
    return out


def create_video_with_plot(parser, metrics):
    unnormalize = UnNormalizer()

    video_dataset = InferenceDataset(parser.video_root,
                                     parser.vid_name,
                                     parser.video_clip_length,
                                     parser.fps,
                                     num_classes=4,
                                     min_side=parser.min_side, 
                                     max_side=parser.max_side,
                                     csv_classes=parser.csv_classes)
    
    video_sampler = SequentialSampler(video_dataset)
    
    video_dataset = DataLoader(video_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=parser.num_workers,
                               sampler=video_sampler)
    
    # create a figure and axis
    fig, ax = plt.subplots()

    # plot the metric data
    line, = ax.plot([], [], 'ro')
    ax.plot(range(len(metrics)), metrics)

    # get the frame rate of the video
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # create a new video writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video = cv2.VideoWriter("output.mp4", fourcc, frame_rate, (640, 480))

    # define a function to update the plot
    def update(time):
        # update the marker on the plot
        line.set_xdata(time)
        line.set_ydata(metrics[time])

    # create the animation using FuncAnimation
    ani = animation.FuncAnimation(fig, update, frames=range(len(metrics)), repeat=False)

    # loop through the frames of the video
    while video.isOpened():
        # read the next frame
        ret, frame = video.read()

        # check if we reached the end of the video
        if not ret:
            break

        # get the current frame number
        frame_number = video.get(cv2.CAP_PROP_POS_FRAMES) - 1

        # update the plot with the current frame
        update(frame_number)

        # draw the plot on the frame
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(image, (640, 480))
        frame[:480, :640] = image

        # write the frame to the output video
        output_video.write(frame)

    # release the video capture and writer
    video.release()
    output_video.release()

