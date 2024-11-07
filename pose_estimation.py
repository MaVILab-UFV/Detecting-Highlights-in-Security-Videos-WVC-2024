import cv2
from ultralytics import YOLO
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json

def are_sholders_between_hands_and_hips(kps):
    """Check if the shoulders are between the hands and hips.

    Args:
        kps: A list of lists of the coordinates of the skeleton points.

    Returns:
        True if the shoulders are between the hands and hips, False otherwise.
    """

    # Get the coordinates of the key points of the hands, shoulders, and hips.
    left_hand_x, left_hand_y = kps[9]
    left_shoulder_x, left_shoulder_y = kps[5]
    left_hip_x, left_hip_y = kps[11]
    right_hand_x, right_hand_y = kps[10]
    right_shoulder_x, right_shoulder_y = kps[6]
    right_hip_x, right_hip_y = kps[12]

    if (right_hand_x == 0 and right_hand_y == 0) and (left_hand_x == 0 and left_hand_y == 0):
        return False
    if(right_hip_x == 0 and right_hip_y == 0) and (left_hip_x == 0 and left_hip_y == 0):
        return False
    if(right_shoulder_x == 0 and right_shoulder_y == 0) and (left_shoulder_x == 0 and left_shoulder_y == 0):
        return False

    shoulder_hand = (left_hand_x - left_shoulder_x, left_hand_y - left_shoulder_y)
    shoulder_hip = (left_hip_x - left_shoulder_x, left_hip_y - left_shoulder_y)

    dot = shoulder_hand[0] * shoulder_hip[0] + shoulder_hand[1] * shoulder_hip[1]

    norm_shoulder_hand = np.sqrt(shoulder_hand[0] ** 2 + shoulder_hand[1] ** 2)
    norm_shoulder_hip = np.sqrt(shoulder_hip[0] ** 2 + shoulder_hip[1] ** 2)

    cos_left = dot / (norm_shoulder_hand * norm_shoulder_hip)

    shoulder_hand = (right_hand_x - right_shoulder_x, right_hand_y - right_shoulder_y)
    shoulder_hip = (right_hip_x - right_shoulder_x, right_hip_y - right_shoulder_y)
    dot = shoulder_hand[0] * shoulder_hip[0] + shoulder_hand[1] * shoulder_hip[1]

    norm_shoulder_hand = np.sqrt(shoulder_hand[0] ** 2 + shoulder_hand[1] ** 2)
    norm_shoulder_hip = np.sqrt(shoulder_hip[0] ** 2 + shoulder_hip[1] ** 2)

    cos_right = dot / (norm_shoulder_hand * norm_shoulder_hip)

    return cos_left < 0 and cos_right < 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,  help='Path to weights file')
    parser.add_argument('--output-folder', type=str,  help='Folder to store output json file')
    parser.add_argument('--input-video', type=str,  help='Path to input video file')
    parser.add_argument('--save-video', action='store_true', help='Save video with the results')

    args = parser.parse_args()

    model = YOLO(args.model_path)
    path = args.input_video

    cap = cv2.VideoCapture(path)

    if args.output_folder is None:
        args.output_folder = ''

    if args.save_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(args.output_folder + 'results.mp4', codec, fps, (width, height))

    hist_pessoa = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        annotated_frame = results[0].plot()
        height, width, _ = annotated_frame.shape

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        kps = results[0].keypoints.xy.cpu().numpy()
        total = len(kps)
        hands_above_shoulder = 0
        if len(kps[0]) != 0 :
            for i, kp in enumerate(kps):
                if are_sholders_between_hands_and_hips(kp):
                    # print(j)
                    bbox = results[0].boxes.xyxy[i].cpu().numpy()
                    confidence = results[0].boxes.conf.cpu().numpy()[i]
                    if(confidence < 0.5):
                        continue
                    confidence_rounded = np.around(confidence, decimals=2)
                    hands_above_shoulder += 1
                    
                    if args.save_video:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f"Hands above shoulders", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            hist_pessoa.append(0)
            if args.save_video:
                output_video.write(frame)
            continue

        hist_pessoa.append(hands_above_shoulder / total)
        if args.save_video:
            output_video.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if args.save_video:
        output_video.release()
    cv2.destroyAllWindows()

    values_pose_estimation = [float(value) for value in hist_pessoa]

    data = {
        "pose": values_pose_estimation
    }
    with open(f"{args.output_folder}pose.json", 'w') as arquivo_json:
        json.dump(data, arquivo_json, indent=4)





