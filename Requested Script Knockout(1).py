import cv2
import pandas as pd
import numpy as np
import os

# Get the directory where this script is running
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script running from: {script_dir}")

def analyze_video(file_path):
    print(f"\nProcessing video: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None

    # Try to open the video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {file_path}")
        return None

    ret, first_frame = cap.read()
    if not ret:
        print(f"[ERROR] Could not read the first frame of: {file_path}")
        cap.release()
        return None

    print("Displaying ROI selection window...")
    
    # Let the user select a region of interest (ROI) for the headgear
    bbox = cv2.selectROI("Select Headgear (Press ENTER to confirm)", 
                        first_frame, 
                        False)
    cv2.destroyWindow("Select Headgear (Press ENTER to confirm)")
    print(f"Selected ROI: {bbox}")

    if bbox == (0, 0, 0, 0):
        print(f"[SKIP] No ROI selected for: {file_path}")
        cap.release()
        return None

    # Initialize CSRT tracker (requires OpenCV contrib module)
    try:
        tracker = cv2.legacy.TrackerCSRT_create()  # Make sure opencv-contrib-python is installed!
    except AttributeError:
        print("[ERROR] TrackerCSRT not available. Install opencv-contrib-python to use this feature.")
        return None

    tracker.init(first_frame, bbox)
    initial_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
    max_distance = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            current_center = (int(bbox[0] + bbox[2] // 2), 
                              int(bbox[1] + bbox[3] // 2))
            distance = np.sqrt((current_center[0] - initial_center[0]) ** 2 + 
                               (current_center[1] - initial_center[1]) ** 2)
            if distance > max_distance:
                max_distance = distance

    cap.release()
    print(f"Finished processing {file_path}. Max distance moved: {max_distance}")
    return max_distance

# Process all videos in the script's directory
results = []
for filename in os.listdir(script_dir):
    if filename.lower().endswith((".mov", ".mp4")):
        print(f"\nFound video file: {filename}")
        file_path = os.path.join(script_dir, filename)
        distance = analyze_video(file_path)
        results.append({
            "Filename": filename,
            "Total Pixels Moved": round(distance, 1) if distance else "ERROR"
        })

# Save results to a CSV file
output_path = os.path.join(script_dir, "knockout_results.csv")
print(f"\nSaving results to: {output_path}")
pd.DataFrame(results).to_csv(output_path, index=False)
print("\nAnalysis complete!")
