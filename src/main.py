import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import json
import time
import sys
import shutil
from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering
import boto3

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'))
if config['offscreen_rendering'] is True:
    vispy.app.use_app('osmesa')
os.makedirs(config['mesh_folder'], exist_ok=True)
os.makedirs(config['video_folder'], exist_ok=True)
os.makedirs(config['depth_folder'], exist_ok=True)
if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
    device = config["gpu_ids"]
else:
    device = "cpu"
print(f"running on device {device}")

sqs = boto3.client("sqs")
s3 = boto3.resource("s3")
s3_uploader = boto3.client("s3")
url = os.environ["AMENA_INPUT_QUEUE_URL"]
input_bucket_name = os.environ["AMENA_INPUT_BUCKET_NAME"]
output_bucket_name = os.environ["AMENA_OUTPUT_BUCKET_NAME"]
s3_input_image_path = "{account_id}/{input_id}/input.jpg"
s3_output_image_path = "{account_id}/{input_id}/output.mp4"
s3_output_error_path = "{account_id}/{input_id}/error.json"
local_input_image_path = config['src_folder'] + "/{account_id}_{input_id}.jpg"
dequeue_wait_time = 20
input_bucket = s3.Bucket(input_bucket_name)

def dequeueInputImage():
    received_queues = []

    while True:
        received_queues = sqs.receive_message(
            QueueUrl=url,
            MaxNumberOfMessages=1,
            VisibilityTimeout=5,
            WaitTimeSeconds=dequeue_wait_time
        )
        if "Messages" in received_queues:
            break

    received_queue = received_queues["Messages"][0]

    queue_id = received_queue["MessageId"]
    receipt_handle = received_queue["ReceiptHandle"]
    queue_body = json.loads(received_queue["Body"])
    queue_type = queue_body["type"]
    account_id = queue_body["account_id"]

    s3_object_key = s3_input_image_path.replace("{account_id}", account_id).replace("{input_id}", queue_id)
    s3_object = input_bucket.Object(s3_object_key)
    img_byte = s3_object.get().get('Body').read()
    img = cv2.imdecode(np.asarray(bytearray(img_byte)), cv2.IMREAD_COLOR)

    return {
        "img": img,
        "account_id": account_id,
        "input_id": queue_id,
        "s3_object": s3_object,
        "sqs_receipt_handle": receipt_handle
    }


while True:
    # Clear work folders.
    if os.path.exists(config['src_folder']):
        shutil.rmtree(config['src_folder'])
    if os.path.exists(config['video_folder']):
        shutil.rmtree(config['video_folder'])
    if os.path.exists(config['depth_folder']):
        shutil.rmtree(config['depth_folder'])
    if os.path.exists(config['mesh_folder']):
        shutil.rmtree(config['mesh_folder'])

    os.mkdir(config['src_folder'])
    os.mkdir(config['video_folder'])
    os.mkdir(config['depth_folder'])
    os.mkdir(config['mesh_folder'])

    # Dequeue request from sqs
    # if failed, continue next
    try:
        dequeued = dequeueInputImage()
    except Exception as e:
        continue

    sqs.change_message_visibility(
        QueueUrl=url,
        ReceiptHandle=dequeued["sqs_receipt_handle"],
        VisibilityTimeout=900
    )

    src_path = local_input_image_path.replace("{account_id}", dequeued["account_id"]).replace("{input_id}", dequeued["input_id"])
    cv2.imwrite(src_path, dequeued["img"])
    print(f"Dequeued: {src_path}")

    s3_output_key = s3_output_image_path.replace("{account_id}", dequeued["account_id"]).replace("{input_id}", dequeued["input_id"])
    s3_error_key = s3_output_error_path.replace("{account_id}", dequeued["account_id"]).replace("{input_id}", dequeued["input_id"])

    try:
        # Get dequeued local input images.
        sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config, config['specific'])
        normal_canvas, all_canvas = None, None

        depth = None
        sample = sample_list[0]
        print("Current Source ==> ", sample['src_pair_name'])
        mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
        image = imageio.imread(sample['ref_img_fi'])

        print(f"Running depth extraction at {time.time()}")
        if config['require_midas'] is True:
            run_depth([sample['ref_img_fi']], config['src_folder'], config['depth_folder'],
                    config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=640)
        if 'npy' in config['depth_format']:
            config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
        else:
            config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
        frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
        config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
        config['original_h'], config['original_w'] = config['output_h'], config['output_w']
        if image.ndim == 2:
            image = image[..., None].repeat(3, -1)
        if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
            config['gray_image'] = True
        else:
            config['gray_image'] = False
        image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
        depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
        mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
            vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
            depth = vis_depths[-1]
            model = None
            torch.cuda.empty_cache()
            print("Start Running 3D_Photo ...")
            print(f"Loading edge model at {time.time()}")
            depth_edge_model = Inpaint_Edge_Net(init_weights=True)
            depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                        map_location=torch.device(device))
            depth_edge_model.load_state_dict(depth_edge_weight)
            depth_edge_model = depth_edge_model.to(device)
            depth_edge_model.eval()

            print(f"Loading depth model at {time.time()}")
            depth_feat_model = Inpaint_Depth_Net()
            depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                        map_location=torch.device(device))
            depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
            depth_feat_model = depth_feat_model.to(device)
            depth_feat_model.eval()
            depth_feat_model = depth_feat_model.to(device)
            print(f"Loading rgb model at {time.time()}")
            rgb_model = Inpaint_Color_Net()
            rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                        map_location=torch.device(device))
            rgb_model.load_state_dict(rgb_feat_weight)
            rgb_model.eval()
            rgb_model = rgb_model.to(device)
            graph = None


            print(f"Writing depth ply (and basically doing everything) at {time.time()}")
            rt_info = write_ply(image,
                                depth,
                                sample['int_mtx'],
                                mesh_fi,
                                config,
                                rgb_model,
                                depth_edge_model,
                                depth_edge_model,
                                depth_feat_model)

            if rt_info is False:
                continue
            rgb_model = None
            color_feat_model = None
            depth_edge_model = None
            depth_feat_model = None
            torch.cuda.empty_cache()
        if config['save_ply'] is True or config['load_ply'] is True:
            verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
        else:
            verts, colors, faces, Height, Width, hFov, vFov = rt_info


        print(f"Making video at {time.time()}")
        videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
        top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
        left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
        down, right = top + config['output_h'], left + config['output_w']
        border = [int(xx) for xx in [top, down, left, right]]
        normal_canvas, all_canvas, video_path = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                            copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(config['video_folder']),
                            image.copy(), copy.deepcopy(sample['int_mtx']), config, image,
                            videos_poses, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                            mean_loc_depth=mean_loc_depth)

        # Put artifacts to s3
        s3_uploader.upload_file(video_path, output_bucket_name, s3_output_key)
        print(f"Put artifacts: {s3_output_key}")

    except Exception as e:
        print("Error: " + repr(e))

        error = { "message": "Unknown error." }

        s3_uploader.put_object(
            Bucket=output_bucket_name,
            Key=s3_error_key,
            Body=json.dumps(error)
        )

        print(f"Put error log: {s3_error_key}")

    finally:
        # delete input source
        sqs.delete_message(
            QueueUrl=url,
            ReceiptHandle=dequeued["sqs_receipt_handle"]
        )

        dequeued["s3_object"].delete()