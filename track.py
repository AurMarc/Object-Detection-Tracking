
import os
import sys
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime
import csv
from analyticss import VehicleAnalytics


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics import YOLO


data_deque = {}
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


class VehicleAnalytics:
    def __init__(self):
        self.frame_data = []
        self.vehicle_directions = defaultdict(list)
        self.zone_entries = defaultdict(dict)
        self.traffic_counts = {'left': 0, 'right': 0}
        self.frame_counts = []
        self.traffic_density = []
        self.vehicle_history = defaultdict(list)
        self.zone_history = defaultdict(lambda: defaultdict(int))
        # Add cumulative counters for vehicles passing through ROIs
        self.roi_passed_vehicles = {'left': set(), 'right': set()}
        self.total_vehicles_passed = {'left': 0, 'right': 0}
        
    def calculate_direction(self, track_id, current_pos, previous_positions):
        if len(previous_positions) < 2:
            return "Initializing"
        
        prev_pos = previous_positions[-2]
        dy = current_pos[1] - prev_pos[1]  
        if dy > 0:
            direction = "South"
        elif dy < 0:
            direction = "North"
        else:
            direction = "Stopped"  

        if track_id not in self.vehicle_directions:
            self.vehicle_directions[track_id] = []
        self.vehicle_directions[track_id].append(direction)
        

        recent_directions = self.vehicle_directions[track_id][-5:]
        if recent_directions:
            return max(set(recent_directions), key=recent_directions.count)
        else:
            return "No Direction"

    def classify_traffic_density(self, vehicles_count, max_capacity=20):
        density = vehicles_count / max_capacity
        if density < 0.3:
            return "Light Traffic"
        elif density < 0.7:
            return "Moderate Traffic"
        else:
            return "Heavy Traffic"
    
    def update_zone_history(self, track_id, zone):
        self.zone_history[track_id][zone] += 1
        
    def get_vehicle_statistics(self, track_id):
        zones = self.zone_history[track_id]
        primary_zone = max(zones.items(), key=lambda x: x[1])[0] if zones else "Unknown"
        directions = self.vehicle_directions[track_id]
        primary_direction = max(set(directions), key=directions.count) if directions else "Unknown"
        return {
            "primary_zone": primary_zone,
            "primary_direction": primary_direction,
            "zone_changes": len(zones)
        }
            
    def generate_insights(self):
        avg_density = np.mean(self.traffic_density) if self.traffic_density else 0
        max_density_frame = np.argmax(self.traffic_density) if self.traffic_density else 0
        
        insights = {
            "total_vehicles_tracked": len(set([data[1] for data in self.frame_data])),
            "peak_vehicle_count": max(self.frame_counts) if self.frame_counts else 0,
            "average_traffic_density": self.classify_traffic_density(avg_density),
            "busiest_frame": max_density_frame,
            "left_lane_total": self.traffic_counts['left'],
            "right_lane_total": self.traffic_counts['right'],
            "zone_statistics": dict(self.zone_history)
        }
        
        return insights

class TrafficAnalyzer:
    def __init__(self, output_path):
        self.analytics = VehicleAnalytics()
        self.output_path = Path(output_path)
        self.csv_path = self.output_path / "traffic_data.csv"
        self.report_path = self.output_path / "traffic_report.txt"
        self.setup_csv()
        
    def setup_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Object_ID', 'Class', 'Direction', 'Zone', 
                           'Left_Current_Count', 'Right_Current_Count',  # Current vehicles in ROI
                           'Left_ROI_Total', 'Right_ROI_Total',         # Cumulative total
                           'Left_Traffic_Status', 'Right_Traffic_Status'])  # Traffic density status
            
    def update_frame_data(self, frame_number, track_id, obj_class, direction, zone, 
                         left_current, right_current):
        # Get cumulative totals
        left_total = len(self.analytics.roi_passed_vehicles['left'])
        right_total = len(self.analytics.roi_passed_vehicles['right'])
        
        # Get traffic status based on current counts
        left_status = self.get_traffic_status(left_current)
        right_status = self.get_traffic_status(right_current)
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_number, 
                track_id, 
                obj_class, 
                direction, 
                zone,
                left_current,
                right_current,
                left_total,
                right_total,
                left_status,
                right_status
            ])
    
    def get_traffic_status(self, count):
        if count > 5:
            return "High"
        elif count >= 3:
            return "Moderate"
        else:
            return "Low"
        
            
    def generate_report(self):
        insights = self.analytics.generate_insights()
        
        with open(self.report_path, 'w') as f:
            f.write("Traffic Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Vehicles Tracked: {insights['total_vehicles_tracked']}\n")
            f.write(f"Peak Vehicle Count: {insights['peak_vehicle_count']}\n")
            f.write(f"Average Traffic Density: {insights['average_traffic_density']}\n\n")
            
            f.write("Lane Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Left Lane Total: {insights['left_lane_total']}\n")
            f.write(f"Right Lane Total: {insights['right_lane_total']}\n\n")
            
            f.write("Traffic Flow Analysis:\n")
            f.write("-" * 20 + "\n")
            if insights['average_traffic_density'] == "Heavy Traffic":
                f.write("- Traffic congestion detected\n")
                if insights['left_lane_total'] > insights['right_lane_total']:
                    f.write("- Left lane experiencing higher traffic volume\n")
                else:
                    f.write("- Right lane experiencing higher traffic volume\n")
            
            f.write("\nRecommendations:\n")
            f.write("-" * 20 + "\n")
            if insights['average_traffic_density'] == "Heavy Traffic":
                if insights['left_lane_total'] > insights['right_lane_total']:
                    f.write("1. Consider traffic signal adjustment for left lane\n")
                    f.write("2. Evaluate possibility of lane expansion in left section\n")
                else:
                    f.write("1. Consider traffic signal adjustment for right lane\n")
                    f.write("2. Evaluate possibility of lane expansion in right section\n")
            f.write("3. Monitor peak traffic times for potential traffic management\n")

def compute_color_for_labels(label):
    """Compute color for different vehicle classes"""
    color_map = {
        2: (222, 82, 175),  # Car
        3: (0, 204, 255),   # Motorcycle
        5: (0, 149, 255),   # Bus
        7: (85, 45, 255)    # Truck
    }
    return color_map.get(label, [int((p * (label ** 2 - label + 1)) % 255) for p in palette])

def calculate_speed(prev_pos, current_pos, fps):
    """Calculate approximate speed of vehicle (pixels per second)"""
    if prev_pos is None:
        return 0
    distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
    return distance * fps / 30  # Normalized by 30 fps

def get_lane_density(count):
    """
    Determine traffic density based on vehicle count in ROI
    count: number of vehicles in the lane's ROI
    returns: (density_label, color)
    """
    if count > 5:
        return "High", (0, 0, 255)  # Red for high traffic
    elif 3 <= count <= 5:
        return "Moderate", (0, 255, 255)  # Yellow for moderate traffic
    else:
        return "Low", (0, 255, 0)  # Green for low traffic



def draw_boxes_and_analyze(frame, bbox_xyxy, identities, object_ids, class_names, roi_y, mid_x, 
                          frame_number, analyzer, fps):
    """Enhanced box drawing with analysis and ROI tracking"""
    height, width = frame.shape[:2]
    left_current = 0  # Current count in left ROI
    right_current = 0  # Current count in right ROI
    current_frame_count = 0
    
    # Create ROI overlays
    left_overlay = frame.copy()
    right_overlay = frame.copy()
    
    # Process each detected object
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
        id = int(identities[i])
        current_frame_count += 1
        
        # Initialize or update tracking
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        data_deque[id].appendleft(center)
        
        # Calculate direction and zone
        direction = analyzer.analytics.calculate_direction(id, center, list(data_deque[id]))
        zone = "Left" if center[0] < mid_x else "Right"
        
        # Check if vehicle is currently in ROI
        if y2 >= roi_y:
            if zone == "Left":
                left_current += 1
            else:
                right_current += 1
            
            # Check if it's a new vehicle passing through ROI
            if id not in analyzer.analytics.roi_passed_vehicles[zone.lower()]:
                analyzer.analytics.roi_passed_vehicles[zone.lower()].add(id)
        
        # Update frame data with both current and total counts
        analyzer.update_frame_data(
            frame_number, 
            id, 
            class_names[object_ids[i]], 
            direction, 
            zone,
            left_current,
            right_current
        )
        
        # Draw visualization
        obj_id = int(object_ids[i])
        color = compute_color_for_labels(obj_id)
        label = f'{id} {class_names[obj_id]} {direction}'
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    
    # Get traffic status
    left_status = analyzer.get_traffic_status(left_current)
    right_status = analyzer.get_traffic_status(right_current)
    
    # Add stats box with both current and total counts
    stats_box = np.zeros((280, 250, 3), dtype=np.uint8)
    cv2.putText(stats_box, f"Frame: {frame_number}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(stats_box, f"Current Total: {current_frame_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Left lane stats
    cv2.putText(stats_box, "Left Lane:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(stats_box, f"Current: {left_current} ({left_status})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(stats_box, f"Total Passed: {len(analyzer.analytics.roi_passed_vehicles['left'])}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Right lane stats
    cv2.putText(stats_box, "Right Lane:", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(stats_box, f"Current: {right_current} ({right_status})", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(stats_box, f"Total Passed: {len(analyzer.analytics.roi_passed_vehicles['right'])}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(stats_box, f"FPS: {int(fps)}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    frame[10:290, 10:260] = stats_box
    
    return frame



def detect(opt):
    # Initialize DeepSORT with optimized parameters
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    cfg.DEEPSORT.MAX_DIST = 0.2
    cfg.DEEPSORT.MIN_CONFIDENCE = 0.3
    cfg.DEEPSORT.MAX_IOU_DISTANCE = 0.7
    cfg.DEEPSORT.MAX_AGE = 70
    cfg.DEEPSORT.N_INIT = 3
    cfg.DEEPSORT.NN_BUDGET = 100
    
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )
    
    # Initialize YOLO model
    model = YOLO(opt.yolo_weights)
    model.fuse()
    model.conf = 0.3
    model.iou = 0.4

    analyzer = TrafficAnalyzer(opt.output)
    frame_count = 0
    
   
    
    # Video capture setup
    cap = cv2.VideoCapture(opt.source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # ROI setup
    roi_height = frame_height // 2
    roi_y = frame_height - roi_height
    mid_x = frame_width // 2
    
    # Video writer setup
    if opt.save_vid:
        save_path = Path(opt.output)
        save_path.mkdir(parents=True, exist_ok=True)
        vid_writer = cv2.VideoWriter(
            str(save_path / "tracked.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
    
    vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1 
        # Draw ROI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, roi_y), (mid_x, frame_height), (0, 255, 0), -1)
        cv2.rectangle(overlay, (mid_x, roi_y), (frame_width, frame_height), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.line(frame, (mid_x, roi_y), (mid_x, frame_height), (255, 255, 255), 2)
        
        # YOLOv11 inference
        results = model.predict(frame, imgsz=opt.imgsz, verbose=False)[0]
        detections = results.boxes.data.cpu().numpy()
        
        if len(detections):
            # Prepare detections for DeepSORT
            bbox_xywh = []
            confs = []
            clss = []
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) in vehicle_classes and conf > model.conf:
                    bbox_xywh.append([
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        x2 - x1,
                        y2 - y1
                    ])
                    confs.append(conf)
                    clss.append(cls)
            
            if bbox_xywh:
                bbox_xywh = torch.Tensor(bbox_xywh)
                confs = torch.Tensor(confs)
                
                # Update tracker
                outputs = deepsort.update(bbox_xywh, confs, clss, frame)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_ids = outputs[:, -1]
                    
                    # Updated drawing and analysis
                    frame = draw_boxes_and_analyze(frame, bbox_xyxy, identities, object_ids, 
                                                 model.names, roi_y, mid_x, frame_count, analyzer,30)                
                
        if opt.show_vid:
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
        if opt.save_vid:
            vid_writer.write(frame)


        analyzer.generate_report()    
    
    # Cleanup
    cap.release()
    if opt.save_vid:
        vid_writer.release()
    if opt.show_vid:
        cv2.destroyAllWindows()

class Opt:
    def __init__(self):
        self.yolo_weights = "yolo11n.pt"
        self.deep_sort_weights = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
        self.source = "inference/output/test3.mp4"
        self.output = "output2nd"
        self.imgsz = 1280
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.show_vid = False
        self.save_vid = True
        self.save_txt = False
        self.config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"
        self.evaluate = False
        self.half = False

if __name__ == '__main__':
    opt = Opt()
    with torch.no_grad():
        detect(opt)

        








