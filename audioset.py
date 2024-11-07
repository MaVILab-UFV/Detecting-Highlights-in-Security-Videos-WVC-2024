import os
import sys
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
from pytorch.models import *    
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import sys
from moviepy.editor import VideoFileClip
import argparse
import json
import csv

labels = ["Shout", "PoliceSiren", "Gunshot"]

with open('class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # # Print audio tagging top probabilities
    # for k in range(10):
    #     print(np.array(labels)[sorted_indexes[k]])
    #     # print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
    #     #     clipwise_output[sorted_indexes[k]]))
    idx = [0,1,2,3]
    for id in idx:
      print('{}: {:.3f}'.format(np.array(labels)[id], 
            clipwise_output[id]))
    


def plot_sound_event_detection_result(framewise_output, file):
    """Visualization of sound event detection result. 

    Args:
      framewise_output: (time_steps, classes_num)
    """
    out_fig_path = f'results/{file}_3.png'
    plt.clf()
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    idx = [0]
    framewise = 0
    print(framewise_output.shape)
    for i in idx:
        framewise += framewise_output[:, i]
    values_gun = framewise
    line, = plt.plot( framewise, label='Disparos de armas de fogo')
    lines.append(line)

    idx = [1]
    framewise = 0
    for i in idx:
        framewise += framewise_output[:, i]
    line, = plt.plot(framewise, label='Gritos')
    lines.append(line)
    values_shouts = framewise
    idx = [2]
    framewise = 0
    for i in idx:
        framewise += framewise_output[:, i]
    line, = plt.plot( framewise, label='Sirene de polícia')
    lines.append(line)
    values_siren = framewise


    plt.legend(handles=lines)
    plt.xlabel('Audio Frames')
    plt.ylabel('Probability')
    plt.yscale('log')
    plt.savefig(out_fig_path)
    print('Save fig to {}'.format(out_fig_path))
    return values_gun, values_shouts, values_siren
    


def sound_event_detection(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, model_type='Cnn14', checkpoint_path=None, device='cuda', cuda=True, interpolate_mode='nearest'):
    """Inference sound event detection result of an audio clip.
    """

    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')

    frames_per_second = sample_rate // hop_size

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    return framewise_output, labels


if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging and sound evetn detection.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, help='Path to input video file')
    parser.add_argument('--output-folder', type=str, help='Folder to store output json file')
    parser.add_argument('--model-path', type=str, help='Path to fine tuned model')

    device = 'cuda' # 'cuda' | 'cpu'
    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = ''

    video_clip = VideoFileClip(args.input_video)  
    video_clip.audio.write_audiofile("aux.mp3") 

    print(labels)
    audio_path = "aux.mp3"
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Sound event detection ------')
    framewise_output, _ = sound_event_detection(
        checkpoint_path=args.model_path,
        device=device, 
        interpolate_mode='nearest', # 'nearest',
        model_type="Cnn14_DecisionLevelMax"
    )
    """(batch_size, time_steps, classes_num)"""

    value_guns, values_shouts, values_siren = plot_sound_event_detection_result(framewise_output, audio_path.split('/')[-1].split('.')[0])
    value_guns = [float(value) for value in value_guns]
    values_shouts = [float(value) for value in values_shouts]
    values_siren = [float(value) for value in values_siren]

    hist_of = [0.2 * values_shouts[i] + 0.2 * values_siren[i] + 0.6 * value_guns[i] for i in range(len(value_guns))]

    with open(f'{args.output_folder}event_audio.json', 'w') as f:
        json.dump({'guns': value_guns, 'shout': values_shouts, 'siren': values_siren, 'val': hist_of}, f)
