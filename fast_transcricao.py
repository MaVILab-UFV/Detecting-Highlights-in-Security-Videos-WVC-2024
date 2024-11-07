from faster_whisper import WhisperModel
import sys
import argparse
from moviepy.editor import VideoFileClip
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, help='Path to input video file')
    parser.add_argument('--output-folder', type=str, help='Folder to store output json file')

    args = parser.parse_args()

    model_size = "large-v3"
    video_path = args.input_video
    video_clip = VideoFileClip(video_path)  
    video_clip.audio.write_audiofile("aux.mp3") 

    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    segments, info = model.transcribe(
        "aux.mp3",
        beam_size=5,
        vad_filter=True,
    )
    if args.output_folder is None:
        args.output_folder = ''

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    if info.language == "en":
        words = [ "run", "gun", "shot",
    "ground", "hands up", "suspect", "reload", "car", "police",
    "drop", "cuffs", "robbery", "victim"]
    else:
        words = ["celular", "perdeu", "mão", "vítima", "roubo", "assalto", "pinote", "passagem", "deita", "viatura"]

    values = []
    for segment in segments:
        keywords = 0
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        for word in words:
            if word in segment.text:
                keywords += 1
        values.append(keywords/len(segment.text.split()))

    data = {
        "keywords": values
    }
    with open(f"{args.output_folder}audio.json", 'w') as arquivo_json:
        json.dump(data, arquivo_json, indent=4)
    