import cv2
import numpy as np

class VisualizationManager:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_height = frame_height // 2
        self.roi_y = frame_height - self.roi_height
        self.mid_x = frame_width // 2

    def draw_roi_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.roi_y), (self.mid_x, self.frame_height), (0, 255, 0), -1)
        cv2.rectangle(overlay, (self.mid_x, self.roi_y), (self.frame_width, self.frame_height), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.line(frame, (self.mid_x, self.roi_y), (self.mid_x, self.frame_height), (255, 255, 255), 2)
        return frame

    def draw_stats_box(self, frame, stats):
        stats_box = np.zeros((280, 250, 3), dtype=np.uint8)
        
        # Draw frame info
        cv2.putText(stats_box, f"Frame: {stats['frame_number']}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(stats_box, f"Current Total: {stats['current_count']}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw lane stats
        for lane in ['left', 'right']:
            y_offset = 70 if lane == 'left' else 140
            cv2.putText(stats_box, f"{lane.capitalize()} Lane:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(stats_box, f"Current: {stats[f'{lane}_current']} ({stats[f'{lane}_status']})", 
                       (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(stats_box, f"Total Passed: {stats[f'{lane}_total']}", 
                       (20, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(stats_box, f"FPS: {int(stats['fps'])}", (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        frame[10:290, 10:260] = stats_box
        return frame