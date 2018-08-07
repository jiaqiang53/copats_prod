# encoding: utf-8
import sys
import cv2
import lib.copats as cpt

conf = cpt.Config()
conf.initialization()


def read_in_video():
    if conf.video_tracking:
        # Read video
        video_path = sys.path[0] + conf.data_path + conf.video_file_name
        video = cv2.VideoCapture(video_path)
    else:
        video = cv2.VideoCapture(conf.camera_index)
    return video


def tracker_runner():
    video = read_in_video()
    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # define an opats tracker
    ot = cpt.OpatsTracker(conf.target_name)
    # port initialization
    if conf.port_write:
        print "Port initialization..."
        sp = cpt.SerialPort(conf.port_name_1, conf.port_name_2, conf.baud_rate)

    # opats controller defination1
    oc = cpt.OpatsPIDController()
    frame_index = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print 'Cannot read video file'
            sys.exit()
        if conf.sub_sample:
            frame = cv2.pyrDown(frame)

        # initialize the tracker
        if ot.initialzing:
            status, bbox = ot.track_target_finder(frame, conf.bbox_width, conf.bbox_height)
            if status and bbox != (0, 0, 0, 0):
                print "#" * 10 + " Target Found!!! Tracking Start!!! " + "#" * 10 + "\n"
                ot.tracker_initialize(frame, bbox)
        # update the tracker
        else:
            ot.tracker_update(frame)
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            frame_index += 1

            # calculation the delta dist through pid controller
            oc.angle_to_command_v1(ot.delta_x, ot.delta_y)

            # write to the port
            if conf.port_write:
                sp.port_writer(angle_x=oc.angle_to_command_x, angle_y=oc.angle_to_command_y)
            print "delta_x: {delta_x}, delta_y: {delta_y}, angle_x: {angle_x}, angle_y: {angle_y}, angle_command_x: " \
                  "{angle_command_x}, angle_command_y: {angle_command_y}".format(delta_x=ot.delta_x,
                                                                                 delta_y=ot.delta_y,
                                                                                 angle_x=oc.angle_x,
                                                                                 angle_y=oc.angle_y,
                                                                                 angle_command_x=oc.angle_to_command_x,
                                                                                 angle_command_y=oc.angle_to_command_y)

    # back to the origin place
    if conf.port_write:
        sp.back_to_origin_poi()


def run():
    tracker_runner()


if __name__ == "__main__":
    run()