# encoding:utf-8
import cv2
import argparse
import sys
import os
import yaml
import math
import cmath
import time
import numpy as np
import serial
import lib.mvsdk as mvsdk

print "Copats is Powered by OpenCV", cv2.__version__

"""
MobileNetSSD_deploy argument
"""
# construct the argument parse
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video",
                    help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt",
                    default="./model/MobileNetSSD_deploy.prototxt",
                    help='Path to text network file: '
                    'MobileNetSSD_deploy.prototxt for Caffe model or ')
parser.add_argument("--weights", default="./model/MobileNetSSD_deploy.caffemodel",
                    help='Path to weights: '
                    'MobileNetSSD_deploy.caffemodel for Caffe model or ')
parser.add_argument("--thr",
                    default=0.2,
                    type=float,
                    help="confidence threshold to filter out weak detections")
args = parser.parse_args()
args.thr = 0.15

# Labels of Network.
classNames = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
               5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
               11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
               16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}


class OpatsTracker:
    def __init__(self, target_name="aeroplane"):
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_index = 0
        self.tracker_type = 'BOOSTING'
        self.track_label = ""
        self.target_name = target_name
        self.bbox = (287, 23, 86, 320)
        self.fps = 0
        self.tracking_status = False
        self.tracker = None
        self.tracking_center_x = 0
        self.tracking_center_y = 0
        self.delta_x = 0
        self.delta_y = 0
        self.frame_shape_half_width = 0
        self.frame_shape_half_height = 0
        self.height_factor = 0
        self.width_factor = 0
        self.net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
        self.initialzing = True

    def tracker_selection(self):
        self.tracker_type = self.tracker_types[self.tracker_index]
        (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
        if int(minor_ver) < 3:
            self.tracker = cv2.Tracker_create(self.tracker_type)
        else:
            if self.tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if self.tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if self.tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if self.tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if self.tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if self.tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()

    def tracker_initialize(self, frame, bbox):
        self.tracker_selection()
        self.bbox = bbox
        self.tracker.init(frame, self.bbox)
        self.frame_shape_half_width = frame.shape[1] / 2
        self.frame_shape_half_height = frame.shape[0] / 2
        self.initialzing = False

    def person_bbox_transfer(self, bbox):
        person_bbox = (bbox[0], bbox[1], bbox[2], bbox[3] / 5)
        return person_bbox

    def track_target_finder(self, frame, bbox_width, bbox_height):
        self.height_factor = frame.shape[0] / 300.0
        self.width_factor = frame.shape[1] / 300.0
        status, label, bbox = self.track_by_dnn_net(frame)
        if status:
            center_x = int(bbox[0] + bbox[2] / 2)
            center_y = int(bbox[1] + bbox[3] / 2)
            bbox = (int(center_x - bbox_width / 2), int(center_y - bbox_height / 2), bbox_width, bbox_height)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 2, 1)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(int(bbox[1]), labelSize[1])
            cv2.putText(frame, label, (int(bbox[0]), yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        cv2.imshow("Targeting", frame)
        return status, bbox

    def tracker_display(self, frame):
        if not self.initialzing:
            # Display tracker type on frame
            cv2.putText(frame, self.tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(self.fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            self.tracker_distance_updater()

            # display the tracking center poi
            cv2.putText(frame, "Position: " + str(self.tracking_center_x) + ', ' + str(self.tracking_center_y), (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # display the delta distance from the tracking center to the img center
            cv2.putText(frame, "Position diff to center: " + str(self.delta_x) + ', ' + str(self.delta_y), (100, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Draw bounding box
        if self.tracking_status:
            # Tracking success
            p1 = (int(self.bbox[0]), int(self.bbox[1]))
            p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            labelSize, baseLine = cv2.getTextSize(self.track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(int(self.bbox[1]), labelSize[1])
            cv2.putText(frame, self.track_label, (int(self.bbox[0]), yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        # Display result
        cv2.imshow("Tracking", frame)

    def tracker_distance_updater(self):
        self.tracking_center_x = int(self.bbox[0]) + int(self.bbox[2]) / 2
        self.tracking_center_y = int(self.bbox[1]) + int(self.bbox[3]) / 2
        self.delta_x = self.tracking_center_x - self.frame_shape_half_width
        self.delta_y = self.frame_shape_half_height - self.tracking_center_y

    def tracker_update(self, frame):
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        self.tracking_status, self.bbox = self.track_by_common_tracker(frame)
        self.track_label = self.target_name + ": LK Tracker"

        net_tracking_status, net_track_label, net_bbox = self.track_by_dnn_net(frame)
        if net_tracking_status:
            self.track_label = net_track_label
            self.bbox = net_bbox

        # Calculate Frames per second (FPS)
        self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # display the tracking status
        self.tracker_display(frame)

    def track_by_common_tracker(self, frame):
        tracking_status, bbox = self.tracker.update(frame)
        return tracking_status, bbox

    def track_by_dnn_net(self, frame):
        frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction
        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size.
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        # Set to network the input blob
        self.net.setInput(blob)
        # Prediction of network
        detections = self.net.forward()

        # Size of frame resize (300x300)
        cols = frame_resized.shape[1]
        rows = frame_resized.shape[0]

        track_status = False
        label = ""
        bbox = (0, 0, 0, 0)

        # For get the class and location of object detected,
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            # Filter prediction
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])  # Class label
                # print class_id, confidence

                if class_id not in classNames:
                    continue

                if classNames[class_id] == self.target_name:
                    print "Get target: %s by dnn net." % self.target_name
                    # Object location
                    xLeftBottom = int(detections[0, 0, i, 3] * cols)
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop = int(detections[0, 0, i, 5] * cols)
                    yRightTop = int(detections[0, 0, i, 6] * rows)

                    # Scale object detection to frame
                    xLeftBottom = int(self.width_factor * xLeftBottom)
                    yLeftBottom = int(self.height_factor * yLeftBottom)
                    xRightTop = int(self.width_factor * xRightTop)
                    yRightTop = int(self.height_factor * yRightTop)

                    track_status = True
                    label = classNames[class_id] + ": dnn_net " + str(confidence)
                    bbox = (xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom)
                    if self.target_name == "person":
                        bbox = self.person_bbox_transfer(bbox)
        return track_status, label, bbox


class OpatsPIDController:
    def __init__(self):
        self.last_delta_angle_x = 1.0
        self.last_delta_angle_y = 1.0
        self.last_angle_x = 1.0
        self.last_angle_y = 1.0
        self.curr_delta_x = 1.0
        self.curr_delta_y = 1.0
        self.cal_cnt = 0
        self.angle_x = 1.0
        self.angle_y = 1.0
        self.angle_to_command_x = 1.0
        self.angle_to_command_y = 1.0
        self.tracking_distance = 200.0
        self.angle_transfer_factor = 360.0 / (2 * cmath.pi)
        self.pid_x_factor = 1.25
        self.pid_y_factor = -0.625
        self.command_period = 50000.0 / 2
        self.resolution = 20000.0
        self.ratio_x = 3.0
        self.ratio_y = 1.0
        # self.pid_p = self.angle_transfer_factor / self.tracking_distance
        self.pid_p_x = 0.45
        self.pid_i_x = 0
        self.pid_d_x = 0.05
        self.pid_p_y = 0.18
        self.pid_i_y = 0
        self.pid_d_y = 0.05

    def pid_calc(self):
        self.curr_delta_angle_x = self.curr_delta_x * 0.001 * 57.3
        self.curr_delta_angle_y = self.curr_delta_y * 0.001 * 57.3

        self.curr_delta_angle_x = self.curr_delta_angle_x - self.last_angle_x / 100 * 15
        self.curr_delta_angle_y = self.curr_delta_angle_y - self.last_angle_y / 100 * 15

        if self.cal_cnt == 0:
            self.angle_x = (self.curr_delta_angle_x * self.pid_p_x) * self.pid_x_factor
            self.angle_y = (self.curr_delta_angle_y * self.pid_p_y) * self.pid_y_factor
        else:
            self.angle_x = (self.curr_delta_angle_x * self.pid_p_x + (self.curr_delta_angle_x - self.last_delta_angle_x) * self.pid_d_x) \
                           * self.pid_x_factor
            self.angle_y = (self.curr_delta_angle_y * self.pid_p_y + (self.curr_delta_angle_y - self.last_delta_angle_y) * self.pid_d_y) \
                           * self.pid_y_factor
        self.cal_cnt += 1
        if self.cal_cnt > 10000:
            self.cal_cnt = 1
        self.last_delta_angle_x = self.curr_delta_angle_x
        self.last_delta_angle_y = self.curr_delta_angle_y
        self.last_angle_x = self.angle_x
        self.last_angle_y = self.angle_y

    def pixel_to_command(self, delta_x, delta_y):
        self.curr_delta_x = delta_x
        self.curr_delta_y = delta_y
        self.pid_calc()
        pulses_per_circle = 20000.0
        self.angle_to_command_x = self.angle_x * pulses_per_circle / 360.0
        self.angle_to_command_y = self.angle_y * pulses_per_circle / 360.0


class LKTrakHelper:
    def __init__(self):
        self.features = []
        self.valid_features = []
        self.x_center = 0
        self.y_center = 0
        self.bbox = (287, 23, 86, 320)
        self.feature_params = dict(maxCorners=700, qualityLevel=0.04, minDistance=5, blockSize=3,
                                   useHarrisDetector=True, k=0.04)

    def valid_feature_extractor_v1(self):
        index_couples = []
        dists = []
        for i in range(0, len(self.features)):
            for j in range(1, len(self.features)):
                if i == j:
                    continue
                dist = math.sqrt((self.features[i][0] - self.features[j][0]) ** 2 + (self.features[i][1] - self.features[j][1]) ** 2)
                index_couples.append((i, j))
                dists.append(dist)
        sortedDistIndex = np.argsort(dists)
        validIndex = sortedDistIndex[0:len(sortedDistIndex) / 2]
        pointIndex = set()
        for index in validIndex:
            pointIndex.add(index_couples[index][0])
            pointIndex.add(index_couples[index][1])

        self.valid_features = []
        self.x_center = 0
        self.y_center = 0
        for index in pointIndex:
            self.valid_features.append(self.features[index])
            self.x_center += self.features[index][0]
            self.y_center += self.features[index][1]
        if len(self.valid_features) > 1:
            self.x_center /= len(self.valid_features)
            self.y_center /= len(self.valid_features)
        self.x_center = np.int(self.x_center)
        self.y_center = np.int(self.y_center)

    def draw_feature(self, frame, features):
        for x, y in features:
            cv2.circle(frame, (x, y), 5, (255, 0, 255), 2)
        if len(features) < len(self.features):
            print "valid lk_feature num: ", len(features)
            cv2.rectangle(frame, (self.bbox[0], self.bbox[1]), (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]),
                          (0, 255, 0), 2, 1)

        cv2.imshow('lk_track', frame)
        cv2.waitKey(1000)

    def bbox_implement(self, width, height):
        up_left_x = self.x_center - width/2
        if up_left_x < 0:
            up_left_x = self.x_center
        up_left_y = self.y_center - height/2
        if up_left_y < 0:
            up_left_y = self.y_center

        self.bbox = (up_left_x, up_left_y, width, height)
        # print "x_center: %d" % self.x_center, "y_center: %d" % self.y_center, "width: %d" % width, "height: %d" % height

    def feature_detect(self, frame, width, height):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
        mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
        mask[:] = 255  # 将mask赋值255也就是算全部图像的角点

        tracks = []
        self.features = []
        t = time.time()
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)  # 像素级别角点检测
        print "GoodFeatureDetect cost time: %d ms" % (time.time() - t)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        for x, y in [tr[-1] for tr in tracks]:
            self.features.append((x, y))

        self.valid_feature_extractor_v1()
        self.bbox_implement(width, height)
        self.draw_feature(frame, self.valid_features)

    def peer_feature_detect(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
        mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
        mask[:] = 255  # 将mask赋值255也就是算全部图像的角点

        tracks = []
        self.features = []
        t = time.time()
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)  # 像素级别角点检测
        print "GoodFeatureDetect cost time: %d ms" % (time.time() - t)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        for x, y in [tr[-1] for tr in tracks]:
            self.features.append((x, y))

        # self.draw_feature(frame, self.features)


class SerialPort:
    def __init__(self, port1="", port2="", baud_rate=115200):
        self.port_name_1 = port1
        self.port_name_2 = port2
        self.baud_rate = baud_rate
        self.port1 = serial.Serial(self.port_name_1, self.baud_rate)
        # self.port2 = serial.Serial(self.port_name_2, self.baud_rate)
        self.acc_angle_x = 0.0
        self.acc_angle_y = 0.0

    def sign(self, x):
        if x >= 0:
            return 0
        else:
            return 1

    def x_y_to_com(self, angle_x, angle_y):
        a = [0x73]
        a.append(int(self.sign(angle_x)))
        a.append(int(abs(angle_x) // 256))
        a.append(int(abs(angle_x) % 256))
        a.append(int(self.sign(angle_y)))
        a.append(int(abs(angle_y) // 256))
        a.append(int(abs(angle_y) % 256))
        a.append(0x65)
        return a

    def port_writer(self, angle_x, angle_y):
        # angle_x = int(angle_x)
        # angle_y = int(angle_y)
        self.acc_angle_x += angle_x
        self.acc_angle_y -= angle_y
        command = self.x_y_to_com(angle_x, angle_y)

        if self.port1.is_open:
            self.port1.flushInput()
            angle_resx = self.port1.write(command)
            print command

    def back_to_origin_poi(self):
        if self.port1.is_open:
            self.port1.flushInput()
            angle_resx = self.port1.write('x' + str(-self.acc_angle_x) + 'n')
        # if self.port2.is_open:
        #     self.port2.flushInput()
        #     angle_resy = self.port2.write('y' + str(-self.acc_angle_y) + 'n')


class MVCamera:
    def __init__(self):
        self.h_camera = 0
        self.p_frame_buffer = None
        self.device_status = False

    def device_preparation(self):
        dev_list = mvsdk.CameraEnumerateDevice()
        no_dev = len(dev_list)
        if no_dev < 1:
            print("No camera device was found!")
            return False

        dev_info = dev_list[0]
        print "device info:", dev_info

        # 打开相机
        self.h_camera = 0
        try:
            self.h_camera = mvsdk.CameraInit(dev_info, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return False

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(self.h_camera)

        # 判断是黑白相机还是彩色相机
        mono_camera = (cap.sIspCapacity.bMonoSensor != 0)
        print("Black && White Camera: {}".format(mono_camera))

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if mono_camera:
            mvsdk.CameraSetIspOutFormat(self.h_camera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.h_camera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(self.h_camera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(self.h_camera, 0)
        mvsdk.CameraSetExposureTime(self.h_camera, 30 * 1000)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.h_camera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        frame_buffer_size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if mono_camera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        self.p_frame_buffer = mvsdk.CameraAlignMalloc(frame_buffer_size, 16)
        print("buffer got")
        self.device_status = True
        return True
    
    def read(self):
        if not self.device_status:
            print "camera preparation failed..."
            return False, 0
        try:
            p_raw_date, frame_head = mvsdk.CameraGetImageBuffer(self.h_camera, 400)
            mvsdk.CameraImageProcess(self.h_camera, p_raw_date, self.p_frame_buffer, frame_head)
            mvsdk.CameraReleaseImageBuffer(self.h_camera, p_raw_date)

            # 此时图片已经存储在p_frame_buffer中，对于彩色相机p_frame_buffer=RGB数据，黑白相机p_frame_buffer=8位灰度数据
            # 把p_frame_buffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self.p_frame_buffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            # frame = frame.reshape((frame_head.iHeight, frame_head.iWidth, 1 if frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            frame = frame.reshape((frame_head.iHeight, frame_head.iWidth, 3))
            print "mvsdk frame got"
            return True, frame
        except mvsdk.CameraException as e:
            print "log error:", e.message
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
            return False, 0

    def close(self):
        # 关闭相机
        mvsdk.CameraUnInit(self.h_camera)
        # 释放帧缓存
        mvsdk.CameraAlignFree(self.p_frame_buffer)


class Config:
    def __init__(self):
        self.data_path = ""
        self.sub_sample = False
        self.lk_track_helper = False
        self.video_tracking = True
        self.use_mvsdk = False
        self.video_file_name = ""
        self.bbox_width = 0
        self.bbox_height = 0
        self.port_write = False
        self.port_name_1 = ""
        self.port_name_2 = ""
        self.baud_rate = 115200
        self.config_params = {}
        self.camera_index = 0
        self.k = 2
        self.target_name = ""

    def get_port_name(self):
        for root, dirs, files in os.walk("/dev"):
            for file in files:
                if "ttyUSB" in file:
                    self.port_write = True
                    break
        self.port_name_1 = "/dev/ttyUSB0"
        # self.port_name_2 = "/dev/ttyUSB1"

    def initialization(self):
        self.get_port_name()
        config_file = open(sys.path[0] + '/config/config.yaml')
        self.config_params = yaml.load(config_file)
        self.data_path = self.config_params["data_path"]
        self.sub_sample = self.config_params["sub_sample"]
        self.lk_track_helper = self.config_params["lk_track_helper"]
        self.video_tracking = self.config_params["video_tracking"]
        self.use_mvsdk = self.config_params["use_mvsdk"]
        self.video_file_name = self.config_params["video_file_name"]
        self.bbox_width = self.config_params["bbox_width"]
        self.bbox_height = self.config_params["bbox_height"]
        self.baud_rate = self.config_params["baud_rate"]
        self.camera_index = self.config_params["camera_index"]
        self.k = self.config_params["k"]
        self.target_name = self.config_params["target_name"]
        # display the configuration
        self.configuration_display()

    def configuration_display(self):
        print "\n" + "#" * 17 + "  Hello COPATS!  " + "#" * 17 + "\n"
        print "Author: Prof. WK, and LZ, YXS, HJQ\n"
        print "Project configuration listed as below: \n"
        key_len = 25
        value_len = 20
        for k, v in self.config_params.items():
            print "#" + ' ' * 2 + k + ' ' * (key_len - len(k) - 2) + "---  " + str(v) + ' ' * (
            value_len - len(str(v)) - 2) + '#'
        print "\n" + "#" * 50 + "\n"


