import logging
import os
import sched
from typing import Union
import json
from flask import Flask, request, jsonify
from google.cloud import storage
import tempfile
import time
import threading
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
#import vm_test_script
#spatiotemporal-open-surgery => scripts => inference.py
#from spatiotemporal-open-surgery.scripts import inference (incorrect way of importing)
#import sys
#sys.path.insert(1, './spatiotemporal-open-surgery/scripts')
import argparse
import inference_full

app = Flask(__name__)

CLOUD_STORAGE_BUCKET = "jack-open-surgery-website"
requestInFlight = 0
def parse_args():
    proj_path = "/home/jgoler/spatiotemporal-open-surgery" #added this
    #main arguments
    parser = argparse.ArgumentParser(description='Surgery Hand and Keypoint Detection on Video')
    parser.add_argument('--vid_name', type=str, default="hyw7Ue6oW8w.mp4")
    parser.add_argument('--directory', type=str, default="") #changing this from "../scripts"
    parser.add_argument('--keypoints', action='store_true')
    parser.add_argument('--whole_dir', action='store_true')
    parser.add_argument('--cfg', type=str, default="/home/jgoler/spatiotemporal-open-surgery/MULTITASK_FILES/KEYPOINTS_FILES/surgery-hand-detection-new/keypoints.yaml")
    parser.add_argument('--multitaskmodel_loc', type=str, default="/home/jgoler/spatiotemporal-open-surgery/models/20210818_finalmultitaskmodel_152_FINAL.pt")
    #smoothing arguments
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--smooth_actions', action='store_true')
    parser.add_argument('--epsilon', type=int, default=100)
    parser.add_argument('--num_smoothing_frames', type=int, default=15)
    parser.add_argument('--percentage_hits', type=float, default=.4)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--video_root', help='Dataset type, must be one of csv or coco.', type=str, default="/tmp")
    parser.add_argument('--viz', help='Dataset type, must be one of csv or coco.', type=bool, default=True)
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', default='data/AVOS/detections/hand_tool_class_names.csv')
    parser.add_argument('--video_clip_length', default=20)
    parser.add_argument('--num_workers', default=4)#4

    parser.add_argument('--fps', default=13)
    parser.add_argument('--min_side', default=608)
    parser.add_argument('--max_side', default=1024)

    ## detection model
    parser.add_argument('--model', help='Path to model (.pt) file.', type=str, default="weights/bi_optimized_model.pt")
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
    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

@app.route('/download_video', methods=['POST'])
def process_video():
    video = json.loads(request.data)
    return jsonify(process_video_internal(video))

def process_video_internal(video):
    
    print(video)
    
    storage_client = storage.Client()

    bucket = storage_client.bucket(CLOUD_STORAGE_BUCKET)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(video["filename"])
    tmp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp")
    blob.download_to_filename(tmp.name + ".mp4") #adding + ".mp4" at the end
    print(tmp.name + ".mp4") #added mp4
    video["tmpfile"] = tmp.name + ".mp4" #added mp4"

    args = parse_args()
    args.vid_name = os.path.basename(tmp.name) + ".mp4" #added mp4
    # keypointsValue = "True" == video["keypointsChoice"]
   # args.keypoints = keypointsValue
    metrics, g_metrics  =  inference_full.run_inference(args)

    blob = bucket.blob(video["filename"] + ".json")
    blob.upload_from_filename(out_json_name)
    blob.make_public()
    json_url = blob.public_url
    blob = bucket.blob(video["filename"] + "_detection.mp4")
    blob.upload_from_filename(out_video_name)
    blob.make_public()
    video_url = blob.public_url
    os.remove(tmp.name)
    os.remove(out_json_name)
    os.remove(out_video_name)
   # return jsonify({'out_json_name': out_json_name, "out_video": out_video})
    return {"json_url": json_url, "video_url": video_url}
   # return {"test": "test value"}

subscriber = pubsub_v1.SubscriberClient()
# The `subscription_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/subscriptions/{subscription_id}`
subscription_path = subscriber.subscription_path("hai-gcp-dexterous", "jack-open-surgery-website-sub")

def shutdown_machine():
    global requestInFlight
    if (requestInFlight == 0):
        os.system("sudo /usr/sbin/shutdown --no-wall now")

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    global requestInFlight 
    print(f"Received {message}.")
    requestInFlight += 1
    print("requestInFlight after increment {}".format(requestInFlight))
    try:
        result = process_video_internal(json.loads(message.data))
        print(result)
        message.ack()
        requestInFlight -= 1
        print("requestInFlight after decrement {}".format(requestInFlight))
# commenting out the below part 1/16/23
       # if (requestInFlight == 0):
        #    scheduler = sched.scheduler(time.time, time.sleep)
         #   e1 = scheduler.enter(3000, 1, shutdown_machine)
          #  scheduler.run()
           # os.system("sudo /usr/sbin/shutdown --no-wall now")
    except BaseException as error:
        print('An exception occurred: {}'.format(error))

flow_control = pubsub_v1.types.FlowControl(max_messages=1)
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback, flow_control=flow_control)
print(f"Listening for messages on {subscription_path}..\n")

def pubsub_listen():
    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()  # Trigger the shutdown.
            streaming_pull_future.result()  # Block until the shutdown is complete.

thread = threading.Thread(target=pubsub_listen)
thread.daemon = True
thread.start()

app.run(host='0.0.0.0', port=5000)
