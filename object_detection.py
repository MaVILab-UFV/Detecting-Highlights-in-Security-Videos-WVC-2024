from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import argparse
import numpy as np
import json

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input-video', type=str, required=True)
    argparser.add_argument('--model-path', type=str, required=True)
    argparser.add_argument('--checkpoint-path', type=str, required=True)
    argparser.add_argument('--output-folder', type=str, required=False)
    argparser.add_argument('--save-video', action='store_true', help='Save video with the results')

    args = argparser.parse_args()
    if args.output_folder is None:
        args.output_folder = ''


    config_file = args.model_path 

    checkpoint_file = args.checkpoint_path 

    model = init_detector(config_file, checkpoint_file, device='cuda:0') 
    input_video_path = args.input_video

    cap = cv2.VideoCapture(input_video_path)

    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(f"{args.output_folder}result.mp4", fourcc, fps, (width, height))

    values_people = []
    values_guns = []
    max_people_inframe = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = inference_detector(model, frame)
        
        cont = 0
        for result in results[0]:
            cont += 1
        values_people.append(cont)
        max_people_inframe = max(max_people_inframe, cont)

        max_conf = 0
        for result in results[1]:
            max_conf = max(max_conf, result[4])
        values_guns.append(max_conf)

        frame = model.show_result(frame, results, score_thr=0.3, show=False)
        if args.save_video:
            out.write(frame)

    if max_people_inframe != 0:
        values_people = np.array(values_people) / max_people_inframe

    data = {
        'people': values_people,
        'guns': values_guns
    }
    with open(f'{args.output_folder}detection.json', 'w') as f:
        json.dump(data, f)
    cap.release()
    if args.save_video:
        out.release()
