# Traffic Analysis System

A comprehensive video analysis system for monitoring and analyzing traffic flow using YOLO object detection and DeepSORT tracking.

## Features

- Real-time vehicle detection and tracking
- Lane-wise traffic density analysis
- Vehicle counting and classification
- Direction and movement pattern analysis
- Automated report generation
- Visual statistics overlay
- CSV data export for further analysis

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd traffic-analysis-system
```

2. Download and setup DeepSORT files:
   - Download the DeepSORT zip file from [here](https://drive.google.com/file/d/1_BYFE6Mt7MxO-VcWt12BT2738QcAlU88/view?usp=sharing)
   - Extract the contents to the `deep_sort_pytorch` folder in the project directory

3. Download input videos:
   - Download the input videos zip file from [here](https://drive.google.com/file/d/15OkDW8Bw607SBD7faD5lMNg8DOnvsfd-/view?usp=sharing)
   - Create an `input` folder in the project directory
   - Extract the videos into the `input` folder

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the analysis:
```bash
python main.py
```

The results will be generated in the `output` folder.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- Deep SORT
- NumPy
- pathlib

## Project Structure

```
├── config.py          # Configuration settings
├── utils.py          # Utility functions for colors and calculations
├── analytics.py      # Traffic analysis and metrics
├── visualization.py  # Visualization tools and overlays
├── detector.py       # YOLO detection and DeepSORT tracking
├── reporter.py       # Report generation and data export
├── main.py          # Main execution script
├── deep_sort_pytorch/# DeepSORT tracking files (downloaded)
├── input/           # Input videos folder (downloaded)
└── output/          # Generated results
```

## Configuration

Key settings in `config.py`:

- `VEHICLE_CLASSES`: Set of vehicle types to detect (car, motorcycle, bus, truck)
- `CONFIDENCE_THRESHOLD`: Minimum detection confidence (default: 0.3)
- `IOU_THRESHOLD`: Intersection over Union threshold (default: 0.4)
- `IMAGE_SIZE`: Input image size for processing (default: 1280)
- `SHOW_VIDEO`: Toggle real-time video display
- `SAVE_VIDEO`: Toggle processed video saving

## Output

### Video Output
- Processed video with tracking overlays
- Bounding boxes around detected vehicles
- Vehicle ID and class labels
- Direction indicators
- Real-time statistics overlay

### Data Export
- Frame-by-frame vehicle counts
- Lane-wise traffic density
- Vehicle trajectories
- Direction patterns

### Analysis Report
- Overall traffic statistics
- Peak traffic periods
- Lane-wise analysis
- Traffic flow patterns
- Recommendations based on analysis

## Advanced Features

### Traffic Density Analysis
- Real-time density classification (Low/Moderate/High)
- Lane-specific monitoring

### Vehicle Tracking
- Unique ID assignment
- Movement pattern analysis
- Zone-based tracking
- Direction classification

### Analytics
- Vehicle count per lane
- Traffic flow patterns
- Peak period identification
- Automated insights generation

## Customization

The system can be customized through `config.py`:

1. Detection Settings:
   - Adjust confidence thresholds
   - Modify vehicle classes
   - Change input image size

2. Tracking Parameters:
   - Maximum detection distance
   - Track age settings
   - IOU thresholds

3. Visualization Options:
   - Toggle video display
   - Modify overlay appearance
   - Adjust ROI settings
