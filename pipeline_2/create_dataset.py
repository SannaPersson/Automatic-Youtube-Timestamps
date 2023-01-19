import pandas as pd
import argparse
import ffmpeg
import torch
import os
import pydub
import multiprocessing
import json
from yt_dlp import YoutubeDL
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from get_timestamps import get_timestamps, get_text, get_timestamp_rows

def split_30_s(audio_file, out_dir):
    # Split the audio file into 30 second chunks
    # Input: audio_file (str): path to the audio file
    # Output: None
    # Side effect: creates a folder vad_chunks and saves the chunks in it

    # Create the vad_chunks folder if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load the audio file
    audio = pydub.AudioSegment.from_wav(audio_file)
    # split audio into 30 second chunks and get start and end time of each chunk
    file_name = audio_file.split("/")[-1].split(".")[0]
    start_time = 0
    for i, chunk in enumerate(audio[::30000]):
        end_time = start_time + len(chunk) / 1000
        # Save the chunk in the vad_chunks folder
        chunk.export(
            f"{out_dir}/{file_name}__{start_time}_{end_time}.wav", format="wav"
        )
        start_time = end_time


def get_timestamps_and_text(urls, descriptions, titles):
    # Extract the timestamps and text from the description
    # Input: descriptions (list): list of descriptions
    # Output: timestamps (list): list of timestamps
    #         text (list): list of text
    timestamps_list = {}
    for url, description, title in zip(urls, descriptions, titles):
        timestamps = get_timestamps(description)
        rows = get_timestamp_rows(description, timestamps)
        if len(rows) == 0 or len(timestamps) == 0:
            continue
        texts = get_text(description, timestamps)
        timestamps_list[url] = {"title": title, "timestamps": texts}
    return timestamps_list


def download_audio(url, output_dir):
    """
    Input: url (str): url of the video
           output_dir (str): path to the output directory
    Output: None
    Side effect: downloads the audio file in the output directory
    """
    if os.path.exists(output_dir + "/" + url.split("=")[-1] + ".wav"):
        print("Already exists")
        return
    # find which subtitles available with listsubtitles
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    # Download audio and caption file, we want specifically english captions and vtt format
    # download in wav format, 16 kHz, a:c to pcm_s16le, normalize audio to -14 dB
    video_id = url.split("=")[-1]

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_dir + "/" + video_id + ".mp3",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "128",
            }
        ],
        "subtitlesformat": "vtt",
        "postprocessor_args": [
            "-ar",
            "16000",
            '-ac', '1',
            #'-acodec', 'pcm_s16le',
            #'-af', 'loudnorm=I=-14:LRA=11:TP=-1.5',
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def preprocess_csv(csv_file, out_csv, out_json):
    # Preprocess the csv file to get the timestamps and text
    # Input: csv_file (str): path to the csv file
    # Output: None
    # Side effect: saves the csv file with the timestamps and text
    df = pd.read_csv(csv_file)
    # drop rows with no description
    df = df.dropna(subset=["description"])
    # drop rows with no timestamps for which get_timestamps returns an empty list
    df = df[df["description"].apply(lambda x: len(get_timestamps(x)) > 0)]

    descriptions = df["description"].tolist()
    urls = df["url"].tolist()
    titles = df["title"].tolist()
    timestamps_list = get_timestamps_and_text(urls, descriptions, titles)
    # save timestamps_list to a json
    with open(out_json, "w") as f:
        json.dump(timestamps_list, f)
    # get urls
    urls = list(timestamps_list.keys())
    print(len(urls))
    # drop rows that are not in urls
    df = df[df["url"].isin(urls)]
    df.to_csv(out_csv, index=False)


def get_audio_files(urls, out_dir):
    # Download the audio from the urls
    # Input: urls (list): list of urls
    # Output: None
    # Side effect: creates a folder audio and saves the audio files in it
    # Create the audio folder if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with Pool(multiprocessing.cpu_count()) as p:
        tqdm(p.map(partial(download_audio, output_dir=out_dir), urls), total=len(urls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        type=str,
        default="youtube_data.csv",
        help="path to the csv file",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="youtube_data_preprocessed.csv",
        help="path to the csv file",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="timestamps_lex.json",
        help="path to the json file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/thedrive/lex_podcasts",
        help="path to the output directory",
    )
    args = parser.parse_args()
    preprocess_csv(args.csv_file, args.out_csv, args.out_json)
    # load timestamps_list.json and get urls
    with open(args.out_json, "r") as f:
        timestamps_list = json.load(f)
    urls = list(timestamps_list.keys())
    # get audio get_audio_files 
    get_audio_files(urls, args.out_dir)
        
