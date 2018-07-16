# encoding:utf-8
import cv2
import sys
import os
import yaml
import cmath
import time
import numpy as np
import serial

print "Copats is Powered by OpenCV", cv2.__version__


class OpatsTracker:
    def __init__(self):
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_index = 0
        self.tracker_type = 'BOOSTING'
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

    def tracker_display(self, frame):
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
        self.tracking_status, self.bbox = self.tracker.update(frame)

        # Calculate Frames per second (FPS)
        self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # display the tracking status
        self.tracker_display(frame)


class OpatsPIDController:
    def __init__(self):
        self.last_delta_x = 1.0
        self.last_delta_y = 1.0
        self.curr_delta_x = 1.0
        self.curr_delta_y = 1.0
        self.cal_cnt = 0
        self.angle_x = 1.0
        self.angle_y = 1.0
        self.angle_to_command_x = 1.0
        self.angle_to_command_y = 1.0
        self.tracking_distance = 2000.0
        self.angle_transfer_factor = 360.0 / (2 * cmath.pi)
        self.pid_x_factor = 0.5
        self.pid_y_factor = 0.3
        self.command_period = 50000.0 / 2
        self.resolution = 20000.0
        self.ratio_x = 3.0
        self.ratio_y = 1.0
        self.pid_p = self.angle_transfer_factor / self.tracking_distance
        self.pid_i = 0
        self.pid_d = 0.01

    def pid_calc(self):
        if self.cal_cnt == 0:
            self.angle_x = (self.curr_delta_x * self.pid_p) * self.pid_x_factor
            self.angle_y = (self.curr_delta_y * self.pid_p) * self.pid_y_factor
        else:
            self.angle_x = (self.curr_delta_x * self.pid_p + (self.curr_delta_x - self.last_delta_x) * self.pid_d) \
                           * self.pid_x_factor
            self.angle_y = (self.curr_delta_y * self.pid_p + (self.curr_delta_y - self.last_delta_y) * self.pid_d) \
                           * self.pid_y_factor
        self.cal_cnt += 1
        self.last_delta_x = self.curr_delta_x
        self.last_delta_y = self.curr_delta_y

    def angle_to_command_v1(self, delta_x, delta_y):
        self.curr_delta_x = delta_x
        self.curr_delta_y = delta_y
        self.pid_calc()
        self.angle_to_command_x = self.angle_x
        self.angle_to_command_y = self.angle_y

    def angle_to_command(self, delta_x, delta_y):
        self.curr_delta_x = delta_x
        self.curr_delta_y = delta_y
        self.pid_calc()
        if self.angle_x == 0:
            self.angle_to_command_x = 1000000.0
        else:
            self.angle_to_command_x = self.command_period / (self.ratio_x * self.angle_x * self.resolution / 360.0)
        if self.angle_y == 0:
            self.angle_to_command_y = 1000000.0
        else:
            self.angle_to_command_y = self.command_period / (self.ratio_y * self.angle_y * self.resolution / 360.0)


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
                dist = cmath.sqrt((self.features[i][0] - self.features[j][0]) ** 2 + (self.features[i][1] - self.features[j][1]) ** 2)
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

    def draw_feature(self, frame):
        print "valid lk_feature num: ", len(self.valid_features)
        for x, y in self.valid_features:
            cv2.circle(frame, (x, y), 5, (255, 0, 255), 2)
        cv2.rectangle(frame, (self.bbox[0], self.bbox[1]), (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]),
                      (0, 255, 0), 2, 1)

        cv2.imshow('lk_track', frame)
        cv2.waitKey(2000)

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
        self.draw_feature(frame)


class SerialPort:
    def __init__(self, port1="", port2="", baud_rate=115200):
        self.port_name_1 = port1
        self.port_name_2 = port2
        self.baud_rate = baud_rate
        self.port1 = serial.Serial(self.port_name_1, self.baud_rate)
        self.port2 = serial.Serial(self.port_name_2, self.baud_rate)
        self.acc_angle_x = 0.0
        self.acc_angle_y = 0.0

    def port_writer(self, angle_x, angle_y):
        angle_x = round(angle_x, 4)
        angle_y = round(angle_y, 4)
        self.acc_angle_x += angle_x
        self.acc_angle_y -= angle_y
        if self.port1.is_open:
            self.port1.flushInput()
            angle_resx = self.port1.write('x' + str(angle_x) + 'n')
        if self.port2.is_open:
            self.port2.flushInput()
            angle_resy = self.port2.write('y' + str(-angle_y) + 'n')

    def back_to_origin_poi(self):
        if self.port1.is_open:
            self.port1.flushInput()
            angle_resx = self.port1.write('x' + str(-self.acc_angle_x) + 'n')
        if self.port2.is_open:
            self.port2.flushInput()
            angle_resy = self.port2.write('y' + str(-self.acc_angle_y) + 'n')


class KMeans:
    def __init__(self, data_set, k):
        self.k = k
        self.data_set= data_set
        self.data_size = len(data_set)
        self.center_set = {}
        self.data_cluster = {}
        self.data_center_dist = {}

    def dist_eclud(self, p1, p2):
        return cmath.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def random_center(self):
        p_min = 100000
        p_max = 0
        for item in self.data_set:
            if item[0] < p_min:
                p_min = item[0]
            if item[0] > p_max:
                p_max = item[0]
            if item[1] < p_min:
                p_min = item[1]
            if item[1] > p_max:
                p_max = item[1]
        centers = (p_max - p_min) * np.random.random_sample((self.k, 2)) + p_min
        index = 1
        for item in centers:
            self.center_set[index] = item
            index += 1

    def update_center(self):
        k_center_list = {}
        for i in range(1, self.k+1):
            k_center_list[i] = []
        for i in range(0, self.data_size):
            k_center_list[self.data_cluster[i]].append(self.data_set[i])
        for i in range(1, self.k+1):
            self.center_set[i] = np.average(k_center_list[i])

    def kmeans(self):
        for i in range(0, self.data_size):
            self.data_center_dist[i] = 10000
            self.data_cluster[i] = 1
        cluster_changed = True
        while cluster_changed:
            cluster_changed = False
            for i in range(0, self.data_size):
                min_index = self.data_cluster[i]
                for center_index, center_point in self.center_set.items():
                    if self.dist_eclud(self.data_set[i], center_point) < self.data_center_dist[i]:
                        self.data_center_dist[i] = self.dist_eclud(self.data_set[i], center_point)
                        min_index = center_index
                if min_index != self.data_cluster[i]:
                    cluster_changed = True
        self.update_center()


class Config:
    def __init__(self):
        self.data_path = ""
        self.sub_sample = False
        self.lk_track_helper = False
        self.video_tracking = True
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

    def get_port_name(self):
        for root, dirs, files in os.walk("/dev"):
            for file in files:
                if "ttyUSB" in file:
                    self.port_write = True
                    break
        self.port_name_1 = "/dev/ttyUSB0"
        self.port_name_2 = "/dev/ttyUSB1"

    def initialization(self):
        self.get_port_name()
        config_file = open(sys.path[0] + '/config/config.yaml')
        self.config_params = yaml.load(config_file)
        self.data_path = self.config_params["data_path"]
        self.sub_sample = self.config_params["sub_sample"]
        self.lk_track_helper = self.config_params["lk_track_helper"]
        self.video_tracking = self.config_params["video_tracking"]
        self.video_file_name = self.config_params["video_file_name"]
        self.bbox_width = self.config_params["bbox_width"]
        self.bbox_height = self.config_params["bbox_height"]
        self.baud_rate = self.config_params["baud_rate"]
        self.camera_index = self.config_params["camera_index"]
        self.k = self.config_params["k"]

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


