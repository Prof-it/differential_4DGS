#
# Copyright (C) 2025, Felix Hirt
# All rights reserved.
#

import os
import glob 
import shutil
from typing import Sequence
from audioalignment import find_time_offset

from absl import flags
from absl import app
from subprocess import Popen, PIPE
__VIDEOSDIR = flags.DEFINE_string(
    name='videosdir',
    default=None,
    help='The directory with the videos to synch',
    required=True)
_REFERENCEVIDEO = flags.DEFINE_string(
    name='referencevideo',
    default=None,
    help='The filepath of the video the synchronization should be based on.',
    required=True)
_TARGETDIR = flags.DEFINE_string(
    name='targetdir',
    default=None,
    help='The target filepath where the extracted files will land',
    required=True)
_FRAMESTART = flags.DEFINE_integer(
    name='framestart',
    default=0,
    help='The frame of the referencevideo where the scene starts')
_ENDFRAME = flags.DEFINE_integer(
    name='endframe',
    default=100,
    help='The frame of the referencevideo where the scene ends')
_FRAMERATE = flags.DEFINE_integer(
    name='framerate',
    default=30,
    help='The framerate of the videos')
_DOWNSCALE = flags.DEFINE_bool(
    name='downscale',
    default=False,
    help='If the videos should be downscaled to 1080p do the interpolation')
_SKIPSYNC = flags.DEFINE_bool(
    name='skipsync',
    default=False,
    help='Only extract images')

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_cam_name(video_path):
    print(os.path.normpath(video_path).split("\\")[-2])
    return os.path.normpath(video_path).split("\\")[-2] + os.path.splitext(os.path.basename(video_path))[0] 

def copyFrameDirAsIs(src, dst):
    dst = os.path.join(dst, "frames", os.path.basename(src))
    print("copying dir as is " + src + " => "+dst )
    if os.path.exists(dst):
        shutil.rmtree(dst)
    try:
        print("copy")
        shutil.copytree(src, dst)
    except:
        raise

def extractFrames(video_path, video_dir, start_frame_num, end_frame_num, image_ext="jpg"):
    """
    Extracts a specified number of frames from a video and saves them as image files.
    
    """
    print("Extracting Frames of " + video_path + " frame " +str(start_frame_num)+ " - " + str(end_frame_num))
    #cam = cv2.VideoCapture(video_path)
    #video_dir = os.path.dirname(video_path)
    save_dir = os.path.join(video_dir, "frames")
    video_counter = 0
    success = True
    # video name
    video_name = get_cam_name(video_path)
    # output dir 
    output_dir = os.path.join(save_dir, video_name)
    if(os.path.isfile(os.path.join(output_dir, str(end_frame_num-start_frame_num-1)+"."+image_ext))):
        print("File exists " + output_dir)
        return output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if(_DOWNSCALE.value):
        args = "ffmpeg", "-i", video_path, "-vf", "select='between(n,"+str(start_frame_num)+","+ str(end_frame_num-1)+")',scale=1920:1080", "-vsync", "0","-qmin", "1", "-qmax", "1", "-start_number", "0", output_dir+"/%d."+image_ext 
    else:
        args = "ffmpeg", "-i", video_path, "-vf", "select='between(n,"+str(start_frame_num)+","+ str(end_frame_num-1)+")'", "-vsync", "0","-qmin", "1", "-qmax", "1", "-start_number", "0", output_dir+"/%d."+image_ext
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    output, err = process.communicate()
    exit_code = process.wait()
    '''
    while video_counter < end_frame_num:
        try:
            save_path = os.path.join(output_dir, f"{(video_counter - start_frame_num)}."+image_ext)
            if (not os.path.exists(save_path)) and (video_counter > start_frame_num - 1):
                ret, frame = cam.read()
                if not ret:
                    print(f"Failed to read frame at {video_counter}. Ending extraction.")
                    break
                cv2.imwrite(save_path, frame)
            video_counter += 1 
        except Exception as e:
            success = False
            print(f"Error while extracting frames: {e}")
            break
    
    cam.release()
    '''
    print(err)
    if success:
        print(f"Successfully extracted {(video_counter-start_frame_num)} frames from {video_path}")
        return output_dir
    else:
        print(f"Failed to extract frames from {video_path}")

def sync():
    print("starting sync")

    #extract frames from reference camera
    referenceFramesDir = extractFrames(_REFERENCEVIDEO.value, _TARGETDIR.value, _FRAMESTART.value, _ENDFRAME.value)

    tmp_audio_dir = os.path.normpath(os.path.join(_TARGETDIR.value, "tmp_audio_files"))
    if not os.path.exists(tmp_audio_dir):
        os.makedirs(tmp_audio_dir)

    tmp_video_dir = os.path.normpath(os.path.join(_TARGETDIR.value, "tmp_video_files"))
    if not os.path.exists(tmp_video_dir):
        os.makedirs(tmp_video_dir)

    #loop through cams
    video_path_list = glob.glob(os.path.join(__VIDEOSDIR.value,"**", "*.mp4"), recursive=True)

    for video_path in video_path_list:

        #if referencevideo continue
        if(video_path==_REFERENCEVIDEO.value):
            continue
        
        #check if endframe already exists and skip this folder
        if(os.path.isfile(os.path.join(_TARGETDIR.value, "frames", get_cam_name(video_path), str(_ENDFRAME.value-_FRAMESTART.value-1)+".png"))):
            print("Skipped folder " + video_path)
            continue

        if(not _SKIPSYNC.value):
            #get audio from reference and 1st cam and get seconds off
            audioresult = find_time_offset([_REFERENCEVIDEO.value, video_path], tmp_audio_dir, [0,0])
            print("Audiooffset: "+ str(audioresult[0][0]))
    
            #extract frames with the rough offset to tmp dir
            roughOffset = round(audioresult[0][0] * _FRAMERATE.value)
            print("RoughFrameOffset: "+ str(roughOffset))
            offsetFrameStart = _FRAMESTART.value + roughOffset
            offsetFrameEnd = _ENDFRAME.value + roughOffset + 1
            if(offsetFrameStart < 0):
                raise ValueError('Video has no content at needed frame start')
        else:
            offsetFrameStart = _FRAMESTART.value
            offsetFrameEnd = _ENDFRAME.value

        extractedFramesWithoutOffsetDir = extractFrames(video_path, _TARGETDIR.value, offsetFrameStart, offsetFrameEnd)

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sync()

    #start SpacetimeGaussians Pre No Prior script without extracting the frames
    args = 'python', 'pre_no_prior_noframes.py', '--videosdir', _TARGETDIR.value, '--startframe', '0', '--endframe', str(_ENDFRAME.value - _FRAMESTART.value), '--refframe', '0'
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    output, err = process.communicate()
    exit_code = process.wait()

    print(output)
    print("done")


if __name__ == '__main__':
  app.run(main)