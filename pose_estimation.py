# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run pose classification and pose estimation."""

import argparse
import logging
import sys
import time,os

import cv2
from ml import Classifier
from ml import Movenet
from ml import MoveNetMultiPose
from ml import Posenet
import utils
from picamera2 import Picamera2

def run(estimation_model: str,
        tracker_type: str,
        classification_model: str,
        label_file: str,
        camera_id: int,
        width: int,
        height: int) -> None:
    """使用 Picamera2 从 CSI 摄像头读取画面并做姿态估计/分类"""

    # 1. 配置并启动 Picamera2
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1)  # 等待自动对焦/曝光

    # 2. 初始化模型
    if estimation_model in ['movenet_lightning', 'movenet_thunder']:
        pose_detector = Movenet(estimation_model)
    elif estimation_model == 'posenet':
        pose_detector = Posenet(estimation_model)
    elif estimation_model == 'movenet_multipose':
        pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
    else:
        sys.exit('ERROR: 不支持的估计模型。')

    # 3. 可视化与 FPS 参数
    counter, fps = 0, 0
    start_time = time.time()
    row_size = 20; left_margin = 24
    text_color = (0, 0, 255); font_size = 1; font_thickness = 1
    fps_avg_frame_count = 10
    keypoint_thresh = 0.1

    # 4. 初始化分类器（如果需要）
    classifier = None
    show_n = 3
    if classification_model:
        classifier = Classifier(classification_model, label_file)
        show_n = min(show_n, len(classifier.pose_class_names))

    # 5. 主循环
    while True:
        frame = picam2.capture_array()  # 获取 RGB ndarray
        if frame is None:
            sys.exit('ERROR: Picamera2 未获取到帧。')
        counter += 1
        img = cv2.flip(frame, 1)

        # 姿态检测
        persons = (pose_detector.detect(img)
                   if estimation_model == 'movenet_multipose'
                   else [pose_detector.detect(img)])
        img = utils.visualize(img, persons)

        # 姿态分类
        if classifier:
            p = persons[0]
            if min(k.score for k in p.keypoints) >= keypoint_thresh:
                probs = classifier.classify_pose(p)
                for i in range(min(show_n, len(probs))):
                    cv2.putText(
                        img,
                        f"{probs[i].label} ({round(probs[i].score,2)})",
                        (left_margin, (i+2)*row_size),
                        cv2.FONT_HERSHEY_PLAIN,
                        font_size,
                        text_color,
                        font_thickness
                    )
            else:
                cv2.putText(
                    img,
                    'Some keypoints are not detected.',
                    (left_margin, 2*row_size),
                    cv2.FONT_HERSHEY_PLAIN,
                    font_size,
                    text_color,
                    font_thickness
                )

        # 计算并显示 FPS
        if counter % fps_avg_frame_count == 0:
            now = time.time()
            fps = fps_avg_frame_count / (now - start_time)
            start_time = now
        cv2.putText(
            img,
            f"FPS = {int(fps)}",
            (left_margin, row_size),
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            text_color,
            font_thickness
        )

        # 显示与退出
        cv2.imshow(estimation_model, img)
        if cv2.waitKey(1) == 27:  # ESC 键
            break

    # 清理
    picam2.stop()
    cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of estimation model.',
      required=False,
      default='movenet_lightning')
  parser.add_argument(
      '--tracker',
      help='Type of tracker to track poses across frames.',
      required=False,
      default='bounding_box')
  parser.add_argument(
      '--classifier', help='Name of classification model.', required=False)
  parser.add_argument(
      '--label_file',
      help='Label file for classification.',
      required=False,
      default='labels.txt')
  parser.add_argument(
      '--cameraId', help='Id of camera (CSI 模块不使用此参数).',
      required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  args = parser.parse_args()

  run(args.model, args.tracker, args.classifier, args.label_file,
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
