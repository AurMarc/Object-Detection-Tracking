from pathlib import Path
import csv

class TrafficReporter:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.csv_path = self.output_path / "traffic_data.csv"
        self.setup_csv()
    
    def setup_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Frame', 'Object_ID', 'Class', 'Direction', 'Zone',
                'Left_Current_Count', 'Right_Current_Count',
                'Left_ROI_Total', 'Right_ROI_Total',
                'Left_Traffic_Status', 'Right_Traffic_Status'
            ])
    
    def update_frame_data(self, frame_number, track_id, obj_class, direction, zone,
                         left_current, right_current, left_total, right_total,
                         left_status, right_status):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_number, track_id, obj_class, direction, zone,
                left_current, right_current, left_total, right_total,
                left_status, right_status
            ])
    
 