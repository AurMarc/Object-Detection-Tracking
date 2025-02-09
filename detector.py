import torch
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import cv2
from pathlib import Path
import torch
from collections import deque
from utils import ColorUtils, TrafficDensityUtils

class TrafficDetector:
    def __init__(self, config, output_path):
        self.config = config
        self.output_path = Path(output_path)
        self.setup_models()
        self.data_deque = {}

    def setup_models(self):
        # Setup YOLO
        self.model = YOLO(self.config.YOLO_WEIGHTS)
        self.model.fuse()
        self.model.conf = self.config.CONFIDENCE_THRESHOLD
        self.model.iou = self.config.IOU_THRESHOLD

        # Setup DeepSORT
        cfg = get_config()
        cfg.merge_from_file(self.config.CONFIG_DEEPSORT)
        for key, value in self.config.DEEPSORT_CONFIG.items():
            setattr(cfg.DEEPSORT, key, value)

        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )

    def process_video(self, video_path, analytics, visualization):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if self.config.SAVE_VIDEO:
            vid_writer = cv2.VideoWriter(
                str(self.output_path / "tracked.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame = visualization.draw_roi_overlay(frame)
            
            # Process frame
            results = self.model.predict(frame, imgsz=self.config.IMAGE_SIZE, verbose=False)[0]
            detections = results.boxes.data.cpu().numpy()
            
            if len(detections):
                processed_frame = self.process_detections(frame, detections, frame_count, 
                                                       analytics, visualization, fps)
                
                if self.config.SAVE_VIDEO:
                    vid_writer.write(processed_frame)
                
                if self.config.SHOW_VIDEO:
                    cv2.imshow('Tracking', processed_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
        
        cap.release()
        if self.config.SAVE_VIDEO:
            vid_writer.release()
        if self.config.SHOW_VIDEO:
            cv2.destroyAllWindows()

    def process_detections(self, frame, detections, frame_count, analytics, visualization, fps):
        bbox_xywh = []
        confs = []
        clss = []
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) in self.config.VEHICLE_CLASSES and conf > self.model.conf:
                bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                confs.append(conf)
                clss.append(cls)
        
        if bbox_xywh:
            bbox_xywh = torch.Tensor(bbox_xywh)
            confs = torch.Tensor(confs)
            
            outputs = self.deepsort.update(bbox_xywh, confs, clss, frame)
            
            if len(outputs) > 0:
                return self.draw_and_analyze(frame, outputs, frame_count, analytics, 
                                          visualization, fps)
        
        # Add default data logging for frames with no detections
        self.reporter.update_frame_data(
            frame_number=frame_count,
            track_id=None,
            obj_class=None,
            direction=None,
            zone=None,
            left_current=0,
            right_current=0,
            left_total=len(analytics.roi_passed_vehicles['left']),
            right_total=len(analytics.roi_passed_vehicles['right']),
            left_status='Low',
            right_status='Low'
        )
        return frame

    def __init__(self, config, output_path, reporter):  # Add reporter parameter
        self.config = config
        self.output_path = Path(output_path)
        self.reporter = reporter  # Store reporter instance
        self.setup_models()
        self.data_deque = {}

    def draw_and_analyze(self, frame, outputs, frame_count, analytics, visualization, fps):
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -2]
        object_ids = outputs[:, -1]
        
        current_frame_count = 0
        left_current = 0
        right_current = 0
        
        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(i) for i in box]
            center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
            id = int(identities[i])
            current_frame_count += 1
            
            if id not in self.data_deque:
                self.data_deque[id] = deque(maxlen=64)
            self.data_deque[id].appendleft(center)
            
            direction = analytics.calculate_direction(id, center, list(self.data_deque[id]))
            zone = "Left" if center[0] < visualization.mid_x else "Right"
            
            if y2 >= visualization.roi_y:
                if zone == "Left":
                    left_current += 1
                else:
                    right_current += 1
                
                if id not in analytics.roi_passed_vehicles[zone.lower()]:
                    analytics.roi_passed_vehicles[zone.lower()].add(id)
            
            # Log data for each detected vehicle
            self.reporter.update_frame_data(
                frame_number=frame_count,
                track_id=id,
                obj_class=self.model.names[int(object_ids[i])],
                direction=direction,
                zone=zone,
                left_current=left_current,
                right_current=right_current,
                left_total=len(analytics.roi_passed_vehicles['left']),
                right_total=len(analytics.roi_passed_vehicles['right']),
                left_status=TrafficDensityUtils.get_lane_density(left_current)[0],
                right_status=TrafficDensityUtils.get_lane_density(right_current)[0]
            )
            
            # Draw detection box and label
            obj_id = int(object_ids[i])
            color = ColorUtils.compute_color_for_labels(obj_id)
            label = f'{id} {self.model.names[obj_id]} {direction}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        
        stats = {
            'frame_number': frame_count,
            'current_count': current_frame_count,
            'left_current': left_current,
            'right_current': right_current,
            'left_status': TrafficDensityUtils.get_lane_density(left_current)[0],
            'right_status': TrafficDensityUtils.get_lane_density(right_current)[0],
            'left_total': len(analytics.roi_passed_vehicles['left']),
            'right_total': len(analytics.roi_passed_vehicles['right']),
            'fps': fps
        }
        
        return visualization.draw_stats_box(frame, stats)            
