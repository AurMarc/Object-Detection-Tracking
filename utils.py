import numpy as np
import cv2
from collections import deque

class ColorUtils:
    PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    COLOR_MAP = {
        2: (222, 82, 175),  # Car
        3: (0, 204, 255),   # Motorcycle
        5: (0, 149, 255),   # Bus
        7: (85, 45, 255)    # Truck
    }
    
    @classmethod
    def compute_color_for_labels(cls, label):
        return cls.COLOR_MAP.get(label, [int((p * (label ** 2 - label + 1)) % 255) for p in cls.PALETTE])

class MotionUtils:
    @staticmethod
    def calculate_speed(prev_pos, current_pos, fps):
        if prev_pos is None:
            return 0
        distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        return distance * fps / 30

class TrafficDensityUtils:
    @staticmethod
    def get_lane_density(count):
        if count > 6:
            return "High", (0, 0, 255)
        elif 3 <= count <= 6:
            return "Moderate", (0, 255, 255)
        else:
            return "Low", (0, 255, 0)