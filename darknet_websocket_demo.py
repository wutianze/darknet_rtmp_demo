from ctypes import *
import subprocess as sp
import base64
import random
import os
import cv2
import time
import darknet
from threading import Thread, enumerate
from queue import Queue
import pynng
import struct
import time
import numpy as np
import io
from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket
from fastapi.templating import Jinja2Templates
from PIL import Image

frame_width =0
frame_height=0
darknet_width=0
darknet_height=0
network, class_names, class_colors=None,None,None

keep_alive = True
frame_queue = Queue(maxsize=5)
darknet_image_queue = Queue(maxsize=5)
detections_queue = Queue(maxsize=5)
process_fps_queue = Queue(maxsize=10)
transfer_time_queue = Queue(maxsize=10)

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

def get_image(frame_queue,darknet_image_queue,transfer_time_queue,input_address):
    global frame_width
    global frame_height
    with pynng.Pair0() as sock:
        sock.listen(input_address)
        while keep_alive:
            msg = sock.recv()
            recv_time = time.time()
            header = msg[0:24]
            hh,ww,cc,tt = struct.unpack('iiid',header)
            transfer_latency =int((recv_time - tt)*1000.0)
            transfer_time_queue.put(transfer_latency)
            if frame_width == 0:
                frame_width = ww
                frame_height = hh
            hh,ww,cc,tt,ss = struct.unpack('iiid'+str(hh*ww*cc)+'s',msg)
            frame_get = np.frombuffer(ss,dtype=np.uint8)
            frame = frame_get.reshape(hh,ww,cc)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put(frame_rgb)

            frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                    interpolation=cv2.INTER_LINEAR)
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
            darknet_image_queue.put(img_for_detect)

def inference(darknet_image_queue, detections_queue, process_fps_queue):
    #while cap.isOpened():
    while keep_alive:
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.2)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        process_fps_queue.put(fps)
        #print("inference FPS: {}".format(fps))
        darknet.free_image(darknet_image)
    #cap.release()

app = FastAPI()
templates = Jinja2Templates(directory=".")
@app.get("/")
def read_root(request: Request):
    yolo_run()
    return templates.TemplateResponse("fast_cam.html", {"request": request})

@app.websocket("/ws")
async def stream_handler(websocket: WebSocket):
    await websocket.accept()
    global keep_alive
    while keep_alive:
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = process_fps_queue.get()
        transfer_latency = transfer_time_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            #cv2.putText(image,"FPS:"+str(fps),(100,80),cv2.FONT_HERSHEY_COMPLEX,2.0,(100,200,200),5)

            img = Image.fromarray(image).resize((400,300))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            payload = {"img": "data:image/png;base64,%s"%base64.b64encode(img_byte_arr.read()).decode(),"process_fps":fps,"transfer_latency":transfer_latency}
            await websocket.send_json(payload)
   

def yolo_run():
    global network
    global class_names
    global class_colors
    global darknet_width
    global darknet_height
    network, class_names, class_colors = darknet.load_network(
            "./cfg/yolov4.cfg",
            "./cfg/coco.data",
            "./yolov4.weights",
            batch_size=1
            )

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    Thread(target=get_image, args=(frame_queue, darknet_image_queue,transfer_time_queue,"tcp://0.0.0.0:13131")).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, process_fps_queue)).start()
