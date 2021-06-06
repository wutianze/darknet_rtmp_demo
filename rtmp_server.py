import subprocess as sp
import cv2
import os
import glob

rtmpUrl = "rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_477702758_77220464&key=828b2eb0c9402a2db9a78f4c3e2296ad&schedule=rtmp&pflag=1"
image = cv2.imread("./data/dog.jpg")
[height,width,pixels] = image.shape

# ffmpeg command
command = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(width, height),
        '-r', "60",
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv', 
        rtmpUrl]

p = sp.Popen(command, stdin=sp.PIPE)

# read webcamera
while(True):
    # write to pipe
    p.stdin.write(image.tostring())
