#!/usr/bin/env python
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

# from .io import IO
import tools
import tools.utils as utils


class PreProcess():
    """
        利用openpose提取自建数据集的骨骼点数据
    """

    def start(self):

        work_dir = 'D:/st-gcn-master'

        ###########################修改处################
        type_number = 8
        gongfu_filename_list = ['CHANGELANES', 'GOSTRAIGHT', 'LEFTWAIT', 'PULLOVER', 'SLOWDOWN', 'STOP', 'TURNLEFT', 'TURNRIGHT']

        #################################################

        for process_index in range(type_number):

            action_filename = gongfu_filename_list[process_index]
            # 标签信息
            labelAction_name = '{}_{}'.format(action_filename,process_index)
            label_no = process_index

            # 视频所在文件夹
            originvideo_file = 'D:/st-gcn-master/dataset/{}/'.format(action_filename)
            # resized视频输出文件夹
            resizedvideo_file = 'D:/st-gcn-master/mydata/training_lib_KTH_cut_5s_resized/resized/{}/'.format(action_filename)

            videos_file_names = os.listdir(originvideo_file)

            # 1. Resize文件夹下的视频到340x256 30fps
            for file_name in videos_file_names:
                video_path = '{}{}'.format(originvideo_file, file_name)
                outvideo_path = '{}{}'.format(resizedvideo_file,
                                              file_name)
                # print("outvideo_path" + outvideo_path)

                writer = skvideo.io.FFmpegWriter(outvideo_path,
                                                 outputdict={'-f': 'mp4', '-vcodec': 'libx264', '-s': '340x256',
                                                             '-r': '30'})
                reader = skvideo.io.FFmpegReader(video_path)
                for frame in reader.nextFrame():
                    writer.writeFrame(frame)
                writer.close()
                print('{} resize success'.format(file_name))

            # 2. 利用openpose提取每段视频骨骼点数据
            resizedvideos_file_names = os.listdir(resizedvideo_file)
            for file_name in resizedvideos_file_names:
                outvideo_path = '{}{}'.format(resizedvideo_file, file_name)

                # openpose = '{}/examples/openpose/openpose.bin'.format(self.arg.openpose)
                # openpose = 'D:/openpose-master/build/x64/Release/OpenPoseDemo.exe'.format(self.arg.openpose)

                openpose = 'D:/openpose-master/build/x64/Release/OpenPoseDemo.exe'

                video_name = file_name.split('.')[0]
                # video_name = file_name
                output_snippets_dir = 'D:/st-gcn-master/mydata/training_lib_KTH_cut_5s_resized/resized/snippets/{}'.format(video_name)
                output_sequence_dir = 'D:/st-gcn-master/mydata/training_lib_KTH_cut_5s_resized/resized/data'
                output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_name)
                # print("output_snippets_dir" + output_snippets_dir)

                label_name_path = '{}/resource/kinetics_skeleton/label_name_action{}.txt'.format(work_dir,process_index)
                with open(label_name_path) as f:
                    label_name = f.readlines()
                    label_name = [line.rstrip() for line in label_name]

                # pose estimation
                openpose_args = dict(
                    video=outvideo_path,
                    write_json=output_snippets_dir,
                    display=0,
                    render_pose=0,
                    model_pose='COCO')
                    # model_pose='BODY_25')
                command_line = openpose + ' '
                command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])
                shutil.rmtree(output_snippets_dir, ignore_errors=True)
                os.makedirs(output_snippets_dir)
                os.system(command_line)

                # pack openpose ouputs
                video = utils.video.get_video_frames(outvideo_path)

                height, width, _ = video[0].shape


                # 这里可以修改label, label_index
                video_info = utils.openpose.json_pack(
                    output_snippets_dir, video_name, width, height, labelAction_name, label_no)

                if not os.path.exists(output_sequence_dir):
                    os.makedirs(output_sequence_dir)

                with open(output_sequence_path, 'w') as outfile:
                    json.dump(video_info, outfile)
                if len(video_info['data']) == 0:
                    print('{} Can not find pose estimation results.'.format(file_name))
                    return
                else:
                    print('{} pose estimation complete.'.format(file_name))


if __name__ == '__main__':
    p=PreProcess()
    p.start()