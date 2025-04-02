import cv2
import numpy as np
import csv
import glob
import os
import shutil

output_folder = "tracked_videos"
csv_filename = "tracking_results.csv"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)
with open(csv_filename, "w", newline="") as csv_file:
    pass

ball_template_paths = glob.glob(os.path.join("templates", "*.jpg")) + glob.glob(os.path.join("templates", "*.png"))
if not ball_template_paths:
    print("No ball template images found in the templates folder.")
    exit()
ball_templates = []
for tpath in ball_template_paths:
    temp = cv2.imread(tpath)
    if temp is None:
        print(f"Warning: Could not load ball template {tpath}.")
        continue
    temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    h, w = temp_gray.shape
    ball_templates.append((temp_gray, w, h))
if not ball_templates:
    print("No valid ball templates loaded. Exiting.")
    exit()

def find_stable_anchor(cap, num_frames=30, max_candidates=50, window_size=40):
    ret, first_frame = cap.read()
    if not ret:
        return None, None
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h_frame, w_frame = first_gray.shape
    candidates = cv2.goodFeaturesToTrack(first_gray, maxCorners=max_candidates, qualityLevel=0.01, minDistance=10)
    if candidates is None:
        return None, first_frame
    candidates = candidates.reshape(-1, 2)
    initial_positions = candidates.copy()
    disp_sum = np.zeros(len(candidates))
    valid_mask = np.ones(len(candidates), dtype=bool)
    for i, cand in enumerate(candidates):
        x, y = cand
        if x - window_size/2 < 0 or x + window_size/2 > w_frame or y - window_size/2 < 0 or y + window_size/2 > h_frame:
            valid_mask[i] = False
    prev_gray = first_gray.copy()
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_candidates, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, candidates.reshape(-1,1,2), None)
        if new_candidates is None:
            break
        new_candidates = new_candidates.reshape(-1, 2)
        valid = (status.flatten() == 1)
        for i in range(len(candidates)):
            if valid[i] and valid_mask[i]:
                new_x, new_y = new_candidates[i]
                if new_x - window_size/2 < 0 or new_x + window_size/2 > w_frame or new_y - window_size/2 < 0 or new_y + window_size/2 > h_frame:
                    valid_mask[i] = False
                else:
                    disp_sum[i] += np.linalg.norm(new_candidates[i] - initial_positions[i])
        candidates = new_candidates
        prev_gray = curr_gray.copy()
        count += 1
    if np.any(valid_mask):
        disp_sum_valid = np.where(valid_mask, disp_sum, np.inf)
        best_idx = np.argmin(disp_sum_valid)
        anchor = initial_positions[best_idx]
        return (int(anchor[0]), int(anchor[1])), first_frame
    else:
        best_idx = np.argmin(disp_sum)
        anchor = initial_positions[best_idx]
        return (int(anchor[0]), int(anchor[1])), first_frame

video_files = glob.glob(os.path.join("data1", "*.mov"))
if not video_files:
    print("No .mov files found in data1 folder.")
    exit()

origin_patches = []
patch_size = 40
for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        continue
    candidate, first_frame = find_stable_anchor(cap, num_frames=30, max_candidates=50, window_size=patch_size)
    if candidate is None:
        print(f"No stable candidate found in {video_path}")
        cap.release()
        continue
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, last_frame = cap.read()
    if not ret:
        cap.release()
        continue
    last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    h_frame, w_frame = last_gray.shape
    cx, cy = candidate
    if cx - patch_size//2 < 0 or cx + patch_size//2 > w_frame or cy - patch_size//2 < 0 or cy + patch_size//2 > h_frame:
        print(f"Candidate in {video_path} does not stay fully in frame in the last frame.")
        cap.release()
        continue
    x1 = max(cx - patch_size//2, 0)
    y1 = max(cy - patch_size//2, 0)
    patch = first_frame[y1:y1+patch_size, x1:x1+patch_size]
    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
        origin_patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
    cap.release()

if not origin_patches:
    print("No origin patches could be extracted from the videos. Exiting.")
    exit()

origin_ref = np.mean(np.stack(origin_patches, axis=0), axis=0)
origin_ref = np.uint8(origin_ref)
origin_ref_path = "origin_ref.jpg"
cv2.imwrite(origin_ref_path, origin_ref)
print(f"Global origin reference image saved as {origin_ref_path}")

origin_template = cv2.imread(origin_ref_path)
if origin_template is None:
    print("Error: Could not load the generated origin reference image.")
    exit()
origin_gray = cv2.cvtColor(origin_template, cv2.COLOR_BGR2GRAY)
origin_h, origin_w = origin_gray.shape

cap_ref = cv2.VideoCapture(video_files[0])
ret, first_frame_ref = cap_ref.read()
cap_ref.release()
if not ret:
    print("Error reading reference video.")
    exit()
first_gray_ref = cv2.cvtColor(first_frame_ref, cv2.COLOR_BGR2GRAY)
result_ref = cv2.matchTemplate(first_gray_ref, origin_gray, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc_ref = cv2.minMaxLoc(result_ref)
global_anchor = np.mean([ (patch.shape[1]//2, patch.shape[0]//2) for patch in origin_patches ], axis=0)
global_anchor = (int(global_anchor[0]), int(global_anchor[1]))
origin_ref_w = origin_w
print(f"Reference origin width: {origin_ref_w}")
print(f"Global anchor (fixed for all videos): {global_anchor}")

with open(csv_filename, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    for video_path in video_files:
        base_name = os.path.basename(video_path)
        csv_writer.writerow([base_name])
        csv_writer.writerow(["frame_number", "ball_center_x", "ball_center_y", "ball_rel_x", "ball_rel_y", "ball_bbox_w", "ball_bbox_h", "total_path_length", "origin_center_x", "origin_center_y", "origin_bbox_w", "origin_bbox_h"])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video_path = os.path.join(output_folder, f"tracked_{base_name}")
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        ret, first_frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from {video_path}")
            cap.release()
            out_writer.release()
            continue
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        best_ball_val = -np.inf
        best_ball_bbox = None
        for temp_gray, tw, th in ball_templates:
            result = cv2.matchTemplate(first_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_ball_val:
                best_ball_val = max_val
                best_ball_bbox = (max_loc[0], max_loc[1], tw, th)
        if best_ball_bbox is None:
            print(f"Could not detect the ball in {base_name}. Skipping video.")
            cap.release()
            out_writer.release()
            continue
        result_origin = cv2.matchTemplate(first_gray, origin_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val_origin, _, max_loc_origin = cv2.minMaxLoc(result_origin)
        origin_bbox = (max_loc_origin[0], max_loc_origin[1], origin_w, origin_h)
        origin_center = (max_loc_origin[0] + origin_w // 2, max_loc_origin[1] + origin_h // 2)
        cv2.rectangle(first_frame, (best_ball_bbox[0], best_ball_bbox[1]), (best_ball_bbox[0]+best_ball_bbox[2], best_ball_bbox[1]+best_ball_bbox[3]), (0, 255, 0), 2)
        cv2.rectangle(first_frame, (origin_bbox[0], origin_bbox[1]), (origin_bbox[0]+origin_bbox[2], origin_bbox[1]+origin_bbox[3]), (255, 0, 0), 2)
        cv2.imshow("Initial Detections", first_frame)
        cv2.waitKey(1000)
        ball_tracker = cv2.TrackerCSRT_create()
        ball_tracker.init(first_frame, best_ball_bbox)
        origin_tracker = cv2.TrackerCSRT_create()
        origin_tracker.init(first_frame, origin_bbox)
        frame_number = 0
        last_ball_bbox = best_ball_bbox
        ball_start_center = None
        max_path_length = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            frame_to_track = frame.copy()
            ball_success, ball_box = ball_tracker.update(frame_to_track)
            origin_success, origin_box = origin_tracker.update(frame_to_track)
            if ball_success:
                x, y, w_box, h_box = [int(v) for v in ball_box]
                ball_center = (x + w_box//2, y + h_box//2)
                if ball_start_center is None:
                    ball_start_center = ball_center
                d = np.linalg.norm(np.array(ball_center) - np.array(ball_start_center))
                max_path_length = max(max_path_length, d)
                last_ball_bbox = (x, y, w_box, h_box)
                cv2.rectangle(frame_to_track, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                cv2.circle(frame_to_track, ball_center, 3, (0, 0, 255), -1)
                cv2.putText(frame_to_track, f"Ball: {ball_center}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(frame_to_track, "Ball tracking failure", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            if origin_success:
                ox, oy, ow, oh = [int(v) for v in origin_box]
                origin_center = (ox + ow//2, oy + oh//2)
                cv2.rectangle(frame_to_track, (ox, oy), (ox+ow, oy+oh), (255, 0, 0), 2)
                cv2.circle(frame_to_track, origin_center, 3, (255, 0, 0), -1)
                cv2.putText(frame_to_track, f"Origin: {origin_center}", (ox, oy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            else:
                cv2.putText(frame_to_track, "Origin tracking failure", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            if ball_success and origin_success:
                raw_rel_x = ball_center[0] - origin_center[0]
                raw_rel_y = ball_center[1] - origin_center[1]
                current_origin_w = ow
                scale = current_origin_w / origin_ref_w
                norm_rel_x = raw_rel_x * (1/scale)
                norm_rel_y = raw_rel_y * (1/scale)
            else:
                norm_rel_x, norm_rel_y = "fail", "fail"
            csv_writer.writerow([base_name, frame_number, ball_center[0] if ball_success else "fail", ball_center[1] if ball_success else "fail", norm_rel_x, norm_rel_y, last_ball_bbox[2] if ball_success else "fail", last_ball_bbox[3] if ball_success else "fail", max_path_length, origin_center[0] if origin_success else "fail", origin_center[1] if origin_success else "fail", ow if origin_success else "fail", oh if origin_success else "fail"])
            cv2.putText(frame_to_track, f"Rel Ball: ({norm_rel_x}, {norm_rel_y})", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            origin_marker_size = 20
            ax, ay = origin_center
            cv2.rectangle(frame_to_track, (ax - origin_marker_size//2, ay - origin_marker_size//2), (ax + origin_marker_size//2, ay + origin_marker_size//2), (255, 0, 0), 2)
            cv2.putText(frame_to_track, "Origin: (0,0)", (ax - origin_marker_size//2, ay - origin_marker_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            out_writer.write(frame_to_track)
        cap.release()
        out_writer.release()
        csv_writer.writerow([])
cv2.destroyAllWindows()
print("Tracking complete. Results saved in", csv_filename, "and tracked videos in", output_folder)
