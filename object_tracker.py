#!/usr/bin/env python

#########################
# yolov3 deepsort
# distance calculate
# lane detection
# 2021/06/21
#########################

import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
import pyshine as ps
import lane_finding

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test_sample.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './data/video/result.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

###################################################
# 1. Yolov3_deepsort
# 1. Distance Calculation
# 2. Warning the collision
###################################################

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        # list_file = open('detection.txt', 'w')
        # frame_index = -1 
    
    init=True
    mtx, dist = lane_finding.distortion_factors()

    ratio = width/200000
    fps = 0.0
    count = 0 
    while True:
        _, img = vid.read()

        # height, width, channel = img.shape
        # print(height, width, channel)fps  = ( fps + (1./(time.time()-t1)) ) / 2

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        t1 = time.time()

        img_out, angle, colorwarp, draw_poly_img = lane_finding.lane_finding_pipeline(img, init, mtx, dist)

        if angle>1.5 or angle <-1.5:
            init=True
        else:
            init=False
        
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

       
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        img=img_out

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    
            ################################
            # Distance Calculation 
            ################################
            #               car :3                bus                truck
            if class_name == "car" or class_name == "bus" or class_name == "truck":
                
                mid_x = (bbox[0]+bbox[2])/2
                mid_y = (bbox[1]+bbox[3])/2
                # apx_distance = round(((1 - (bbox[2]*ratio - bbox[0]*ratio))**4),1)
                apx_distance = round((((height-bbox[3]))*ratio)*4.5,1)
                #cv2.putText(img, '{:.2f}'.format(apx_distance), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                ps.putBText(img, '{:.2f}'.format(apx_distance), text_offset_x=int(mid_x), text_offset_y=int(mid_y), vspace=1,
                             hspace=1, font_scale=0.7, background_RGB=(228, 225, 222), text_RGB=(1, 1, 1))

                if apx_distance <= 1:
                    if (mid_x) > width*0.3 and (mid_x) < width*0.7:
                        # cv2.putText(img, 'WARNING!!!', (400,150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 4)
                        # cv2.putText(img,  class_name + str(track.track_id), (400,220), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)
                        ps.putBText(img, 'WARNING!!! : '+class_name + str(track.track_id), 400, 150, vspace=1,
                             hspace=1, font_scale=3, background_RGB=(228, 225, 222), text_RGB=(255, 0, 0))

        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (40, 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.namedWindow('output',cv2.WINDOW_NORMAL) # WINDOW_NORMAL
        # cv2.resizeWindow('output',1200, 600)
        cv2.imshow('output', img)

        if FLAGS.output:
            out.write(img)
            # frame_index = frame_index + 1
            # list_file.write(str(frame_index)+' ')
            # if len(converted_boxes) != 0:
            #     for i in range(0,len(converted_boxes)):
            #         list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            # list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    if FLAGS.ouput:
        out.release()
        # list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
