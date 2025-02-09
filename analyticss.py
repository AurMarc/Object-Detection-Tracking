from collections import defaultdict
import numpy as np
import csv
import pandas as pd

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
        self.roi_passed_vehicles = {'left': set(), 'right': set()}
        self.total_vehicles_passed = {'left': 0, 'right': 0}

    def calculate_direction(self, track_id, current_pos, previous_positions):
        if len(previous_positions) < 2:
            return "Initializing"
        
        prev_pos = previous_positions[-2]
        dy = current_pos[1] - prev_pos[1]
        direction = "South" if dy > 0 else "North" if dy < 0 else "Stopped"
        
        self.vehicle_directions[track_id].append(direction)
        recent_directions = self.vehicle_directions[track_id][-5:]
        
        return max(set(recent_directions), key=recent_directions.count) if recent_directions else "No Direction"

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
        
        return {
            "total_vehicles_tracked": len(set([data[1] for data in self.frame_data])),
            "peak_vehicle_count": max(self.frame_counts) if self.frame_counts else 0,
            "average_traffic_density": self.classify_traffic_density(avg_density),
            "busiest_frame": max_density_frame,
            "left_lane_total": self.traffic_counts['left'],
            "right_lane_total": self.traffic_counts['right'],
            "zone_statistics": dict(self.zone_history)
        }
    
    def group_by_frame(self, output_path):
        file_path = output_path / "traffic_data.csv"  # Replace with your actual CSV file path
        df = pd.read_csv(file_path)
        
        # Group by Frame and process required columns
        grouped = df.groupby('Frame').agg({
            'Object_ID': lambda x: ', '.join(map(str, sorted(x.unique()))),  # Unique Object IDs, comma-separated
            'Class': lambda x: ', '.join(sorted(x.dropna().astype(str).unique())),  # Convert to string and handle NaN
            'Left_Current_Count': 'last',  # Last value in the frame
            'Right_Current_Count': 'last',  # Last value in the frame
            'Left_ROI_Total': 'last',  # Last value in the frame
            'Right_ROI_Total': 'last',  # Last value in the frame
            'Left_Traffic_Status': 'last',  # Last value in the frame
            'Right_Traffic_Status': 'last',  # Last value in the frame
        }).reset_index()
        
        # Save the output to a new CSV file
        output_file = output_path / "grouped_by_frame.csv"
        grouped.to_csv(output_file, index=False)
    
        return output_file               
    
    def generate_business_insights(self, csv_file, output_report):
        # Load the CSV file
        df = pd.read_csv(csv_file)
    
        # Calculate overall traffic density for each frame
        traffic_density = df.groupby('Frame').agg({
            'Left_Traffic_Status': 'last',
            'Right_Traffic_Status': 'last'
        }).reset_index()
    
        # Overall summary
        total_frames = df['Frame'].nunique()
        left_heavy_traffic = traffic_density['Left_Traffic_Status'].value_counts().get('High', 0)
        right_heavy_traffic = traffic_density['Right_Traffic_Status'].value_counts().get('High', 0)
        
        # Identify timestamps for heavy traffic
        heavy_traffic_zones = traffic_density[
            (traffic_density['Left_Traffic_Status'] == 'High') | (traffic_density['Right_Traffic_Status'] == 'High')
        ]
    
        # Create recommendations
        recommendations = []
        for _, row in heavy_traffic_zones.iterrows():
            if row['Left_Traffic_Status'] == 'High':
                recommendations.append(f"Heavy traffic detected in Left Zone at Frame {row['Frame']}. Suggest traffic signal adjustment.")
            if row['Right_Traffic_Status'] == 'High':
                recommendations.append(f"Heavy traffic detected in Right Zone at Frame {row['Frame']}. Suggest traffic signal adjustment.")
    
        # Generate the report
        with open(output_report, 'w') as report:
            report.write("Simple Business Insights Report\n")
            report.write("----------------------------------\n")
            report.write(f"Total Frames Analyzed: {total_frames}\n")
            report.write(f"Total Instances of Heavy Traffic in Left Zone: {left_heavy_traffic}\n")
            report.write(f"Total Instances of Heavy Traffic in Right Zone: {right_heavy_traffic}\n\n")
            report.write("Recommendations:\n")
            if recommendations:
                for rec in recommendations:
                    report.write(f"- {rec}\n")
            else:
                report.write("- No heavy traffic detected. Traffic flow appears normal.\n")
        
        print(f"Report saved to {output_report}")    
    