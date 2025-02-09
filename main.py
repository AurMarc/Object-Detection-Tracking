from config import Config
from analyticss import VehicleAnalytics
from reporter import TrafficReporter
from detector import   TrafficDetector
from visualization import VisualizationManager
import cv2
from pathlib import Path
import torch

def main():
    # Setup configuration
    config = Config()
    config.setup_environment()

    input_video_path = "input/vehicles.mp4"
    
    # Initialize components
    output_path = Path("output")
    output_path.mkdir(parents=True, exist_ok=True)
    
    analytics = VehicleAnalytics()
    reporter = TrafficReporter(output_path)
    detector = TrafficDetector(config, output_path, reporter)  
    
    # Setup video capture to get dimensions
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    visualization = VisualizationManager(frame_width, frame_height)
    
    # Process video
    with torch.no_grad():
        detector.process_video(input_video_path, analytics, visualization)
    
    csv_file = analytics.group_by_frame(output_path)
    output_report = output_path/"traffic_insights_report.txt"

    analytics.generate_business_insights(csv_file, output_report)

if __name__ == '__main__':
    main()