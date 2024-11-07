import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

def interpolacao(hist, n):
    hist = np.array(hist)
    x = np.linspace(0, len(hist)-1, n)
    return np.interp(x, np.arange(len(hist)), hist)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output-folder', type=str, required=False)
    args = argparser.parse_args()
    if args.output_folder is None:
        args.output_folder = ''

    with open(f'{args.output_folder}detection.json', 'r') as f:
        data_detection = json.load(f)

    values_people = data_detection['people']
    values_guns = data_detection['guns']

    with open(f'{args.output_folder}pose.json', 'r') as f:
        data_pose = json.load(f)

    values_pose = data_pose['pose']

    with open(f'{args.output_folder}event_audio.json', 'r') as f:
        data_sounds = json.load(f)

    values_event_audio = data_sounds['val']

    with open(f'{args.output_folder}audio.json', 'r') as f:
        data_transcription = json.load(f)

    values_audio = data_transcription['keywords']

    n = len(values_people)
    values_event_audio = interpolacao(values_event_audio, n)
    values_audio = interpolacao(values_audio, n)


    hist_real = [max(values_people[i], values_guns[i], values_pose[i], values_audio[i]) for i in range(n)]
    hist_real = np.array(hist_real)
    hist_real = hist_real.astype(float)

    num_frames = len(hist_real)

    fps = 30
    time_seconds = np.arange(num_frames) / fps

    plt.bar(time_seconds, hist_real, width=0.03) 

    plt.xlabel('Time (seconds)')
    plt.ylabel('Value inferred of relevant moment (0 a 1)')
    plt.ylim(0, 1) 

    plt.savefig(f'{args.output_folder}semantic_profile.png')



