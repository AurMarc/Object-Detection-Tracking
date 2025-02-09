import os
import torch

class Config:
    # Environment settings
    ENV_SETTINGS = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1"
    }
    
    # Model settings
    YOLO_WEIGHTS = "yolo11n.pt"
    DEEP_SORT_WEIGHTS = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    CONFIG_DEEPSORT = "deep_sort_pytorch/configs/deep_sort.yaml"
    
    # Detection settings
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.4
    
    # DeepSORT settings
    DEEPSORT_CONFIG = {
        "MAX_DIST": 0.2,
        "MIN_CONFIDENCE": 0.3,
        "MAX_IOU_DISTANCE": 0.7,
        "MAX_AGE": 70,
        "N_INIT": 3,
        "NN_BUDGET": 100
    }
    
    # Video settings
    IMAGE_SIZE = 1280
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SHOW_VIDEO = False
    SAVE_VIDEO = True

    @classmethod
    def setup_environment(cls):
        for key, value in cls.ENV_SETTINGS.items():
            os.environ[key] = value