import cv2
import pandas as pd
import numpy as np
import os
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script running from: {script_dir}")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Invalid number. Please enter a numeric value.")

def analyze_video(file_path):
    print(f"\nProcessing: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {file_path}")
        return None

    ret, first_frame = cap.read()
    if not ret:
        print(f"[ERROR] Could not read first frame: {file_path}")
        cap.release()
        return None

    print("Select the helmet in the first frame")
    bbox = cv2.selectROI("Select Helmet (Press ENTER)", first_frame, False)
    cv2.destroyWindow("Select Helmet (Press ENTER)")
    print(f"Selected ROI: {bbox}")

    if bbox == (0, 0, 0, 0):
        print(f"[SKIP] No ROI selected for: {file_path}")
        cap.release()
        return None

    # Get required measurements
    print("\nEnter physical measurements:")
    top_cm = get_float("Distance from helmet TOP to roof (cm): ")  # Vertical space above helmet
    bottom_cm = get_float("Distance from helmet BOTTOM to floor (cm): ")  # Vertical space below helmet
    helmet_height_cm = get_float("Actual height of the helmet (cm): ")  # Real-world helmet height

    # Calculate pixel/cm ratio from helmet height
    roi_height_px = bbox[3]  # Height of selected ROI in pixels
    if roi_height_px == 0 or helmet_height_cm <= 0:
        print("[ERROR] Invalid helmet height measurement")
        return None

    # px_to_cm: Conversion factor from pixels to centimeters
    # Calculated as real helmet height divided by pixel height in ROI
    px_to_cm = helmet_height_cm / roi_height_px
    print(f"Pixel-to-cm ratio: {px_to_cm:.4f} px/cm")

    # Pendulum length: Distance from roof to helmet's center of mass
    # Calculated as: roof-to-helmet-top + half the helmet height
    pendulum_length_cm = top_cm + (helmet_height_cm / 2)
    print(f"Pendulum length: {pendulum_length_cm:.1f} cm")

    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        print("[ERROR] TrackerCSRT unavailable. Install opencv-contrib-python.")
        return None

    tracker.init(first_frame, bbox)
    initial_center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
    max_horizontal_px = 0  # Track maximum horizontal movement in pixels

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            current_center = (int(bbox[0] + bbox[2]//2), 
                              int(bbox[1] + bbox[3]//2))
            dx = current_center[0] - initial_center[0]  # Horizontal displacement
            horizontal_dist_px = abs(dx)
            
            if horizontal_dist_px > max_horizontal_px:
                max_horizontal_px = horizontal_dist_px

    cap.release()

    # Convert pixel measurements to real-world centimeters
    max_horizontal_cm = max_horizontal_px * px_to_cm  # Maximum horizontal movement in cm
    
    # Swing percentage: (horizontal movement / pendulum length) * 100
    # Represents how much of a 90° swing was achieved (100% = horizontal position)
    max_swing_percent = (max_horizontal_cm / pendulum_length_cm) * 100
    max_swing_percent = min(max_swing_percent, 100)  # Cap at 100% (90° swing)

    print(f"\nResults for {file_path}:")
    print(f"  Max horizontal movement: {max_horizontal_cm:.1f} cm")
    print(f"  Swing percentage: {max_swing_percent:.1f}%")
    print(f"  Total vertical clearance: {top_cm + bottom_cm + helmet_height_cm:.1f} cm")

    return {
        "max_cm": max_horizontal_cm,
        "percentage": max_swing_percent,
        "pendulum_length": pendulum_length_cm,
        "px_to_cm": px_to_cm,
        "helmet_height_cm": helmet_height_cm
    }

results = []
for filename in os.listdir(script_dir):
    if filename.lower().endswith((".mov", ".mp4")):
        print(f"\nFound video: {filename}")
        file_path = os.path.join(script_dir, filename)
        result = analyze_video(file_path)
        
        if result:
            results.append({
                "Filename": filename,
                # Maximum horizontal movement in centimeters
                "Max Horizontal CM": round(result["max_cm"], 1),
                # Swing percentage (0% = vertical, 100% = horizontal)
                "Swing Percentage": round(result["percentage"], 1),
                # Length from roof to helmet's center of mass (cm)
                "Pendulum Length (CM)": round(result["pendulum_length"], 1),
                # Pixels per centimeter conversion ratio
                "PX/CM Ratio": round(result["px_to_cm"], 4),
                # Actual physical height of the helmet (cm)
                "Helmet Height (CM)": round(result["helmet_height_cm"], 1)
            })
        else:
            # Error handling for failed processing
            results.append({
                "Filename": filename,
                "Max Horizontal CM": "ERROR",
                "Swing Percentage": "ERROR",
                "Pendulum Length (CM)": "ERROR",
                "PX/CM Ratio": "ERROR",
                "Helmet Height (CM)": "ERROR"
            })

output_path = os.path.join(script_dir, "knockout_results.csv")
print(f"\nSaving results to: {output_path}")
pd.DataFrame(results).to_csv(output_path, index=False)
print("\nAnalysis complete!")
