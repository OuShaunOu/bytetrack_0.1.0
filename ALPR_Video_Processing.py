import cv2
import numpy as np
import ultralytics
from dataclasses import dataclass
from yolox.tracker.byte_tracker import BYTETracker, STrack
import torch
from onemetric.cv.utils.iou import box_iou_batch
import pyautogui

yolo_model = "Yolo/yolov8/yolov8n.pt"
yolo_model_version = 'v8'
video_path = 'C:\\Users\\Shaun Johnson\\Desktop\\Work\\Projects\\ALPR\\test_video.mp4' #cars
# video_path = 'C:\\Users\\Shaun Johnson\\Desktop\\Work\\Projects\\ALPR\\ByteTrack\\videos\\palace.mp4' #people
classes_to_detect = [2, 3, 5, 7]  #vehicles
# classes_to_detect = [0]  #person
confidence_thres = 0.6

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = confidence_thres
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# @dataclass
# class Detection:
#     x1: int
#     y1: int
#     x2: int
#     y1: int
#     conf: float
#     class_id: int
#     missing: int

    # @classmethod
    # def from_results(cls, prediction, names: Dict[int, str]):
    #     result = []
    #     for x1, y1, x2, y2, conf, class_id in prediction:
    #         class_id=int(class_id)


    #         result.append(Detection(
    #             rect=Rect(
    #                 x=float(x_min),
    #                 y=float(y_min),
    #                 width=float(x_max - x_min),
    #                 height=float(y_max - y_min)
    #             ),
    #             class_id=class_id,
    #             class_name=names[class_id],
    #             confidence=float(confidence)
    #         ))
    #     return result

class Object_Detection_Tracking:
    def __init__(self, yolo_model, yolo_model_version, video_path, BYTETrackerArgs, classes_to_detect = None) -> None:
        self.yolo_model = yolo_model
        self.video_path = video_path
        self.classes_to_detect = classes_to_detect
        self.yolo_model_version = yolo_model_version     
        self.BYTETrackerArgs = BYTETrackerArgs 


    @staticmethod
    def get_coordinates(detections) -> np.ndarray:
        return np.array(detections[0].boxes.xyxy, dtype=float)
    
    @staticmethod
    def get_conf(detections) -> np.ndarray:
        return np.array(detections[0].boxes.conf, dtype=float)
    
    @staticmethod
    def get_object_ids(output_stracks):
        Objects_id = []
        for i in range(len(output_stracks)):
            Objects_id.append(output_stracks[i].track_id)
        return Objects_id
    
    @staticmethod    
    def Anotate_Frame(frame, coordinates, Objects_id, draw_box = True, id_object = True, Draw_label = False, display_confidence_score = True):
        for i, coordinate in enumerate(coordinates):

            x1, y1 = int(coordinate[0]), int(coordinate[1])
            x2, y2 = int(coordinate[2]), int(coordinate[3])

            if draw_box:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)

            if id_object:
                obj_id = Objects_id[i]
                text = f"{obj_id}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=5)

            if Draw_label:
                print("The draw label feature has not been implemented yet")

            if display_confidence_score:
                conf = round(coordinate[-1], 2)
                text = f"{conf}"
                cv2.putText(frame, text, (x2 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=5)
        return frame

    @staticmethod
    def tracks2boxes(tracks) -> np.ndarray:
        return np.array([
            track.tlbr
            for track
            in tracks
        ], dtype=float)
    
    @staticmethod
    def match_detections_with_tracks(detections, tracks):
        list_to_return = []
        ordered_tracking_ids = []
        detection_boxes = Object_Detection_Tracking.get_coordinates(detections=detections)
        coordinates = detection_boxes.tolist()
        conf = Object_Detection_Tracking.get_conf(detections=detections)
        confidence = conf.tolist()
        for i, lst in enumerate(coordinates):
            temp = lst[:]
            temp.append(confidence[i])
            list_to_return.append(temp)

        print(list_to_return)
        tracks_boxes = Object_Detection_Tracking.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detection_boxes)
        
        track2detection = np.argmax(iou, axis=1)
        
        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                ordered_tracking_ids.append(tracks[tracker_index].track_id)

        return list_to_return, ordered_tracking_ids #this will be a list of lists. Where each list will consist of teh detection_boxes information(x1, y1...), the confidence score of the detection, and finally the object ID
    
    def TrackObject(self, confidence_thres, Show_Frames = True):  
        screen_width, screen_height = pyautogui.size()
        Dictionary_of_relevant_objects = {}

        model = ultralytics.YOLO(self.yolo_model, self.yolo_model_version)

        byte_tracker = BYTETracker(self.BYTETrackerArgs) 
        cap = cv2.VideoCapture(video_path)
        success, Initial_Image = cap.read() 
        # try:            
        img_shape = (list(Initial_Image.shape))[0: 2]

        while cap.isOpened():
            success, frame = cap.read()             

            while not success:
                success, frame = cap.read()

            prediction = model.predict(source=frame, conf = confidence_thres, classes = self.classes_to_detect)

            if len(prediction[0].boxes.xyxy != 0):
                coordinates = Object_Detection_Tracking.get_coordinates(prediction)
                conf = Object_Detection_Tracking.get_conf(prediction)
                
                update_detector = np.concatenate((coordinates, conf.reshape((-1, 1))), axis=1)

                output_stracks = byte_tracker.update(update_detector, img_shape, img_shape) 
                # for element in output_stracks:
                #     print(element)
                #     print(dir(element))
                Objects_id = Object_Detection_Tracking.get_object_ids(output_stracks) 

                list_to_return, ordered_tracking_ids = Object_Detection_Tracking.match_detections_with_tracks(detections=prediction, tracks=output_stracks)

                # print("This is maybe the better option for prediction structure", test)

                if len(ordered_tracking_ids) == len(list_to_return):
                    frame = Object_Detection_Tracking.Anotate_Frame(frame, list_to_return, ordered_tracking_ids)
                else:
                    pass
            if Show_Frames:
                frame = cv2.resize(frame, (screen_width, screen_height))
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
        # except:
        #     print("An error occured while analysing the video")

Instance = Object_Detection_Tracking(yolo_model, yolo_model_version, video_path, BYTETrackerArgs, classes_to_detect=classes_to_detect)
Instance.TrackObject(confidence_thres) 


