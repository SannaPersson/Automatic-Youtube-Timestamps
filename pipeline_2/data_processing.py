import json
import argparse
from yt_dlp import YoutubeDL
import os
import pydub
import pandas as pd
from tqdm import tqdm
from csv import writer


def split_30_s(audio_file, out_dir):
    # Split the audio file into 30 second chunks
    # Input: audio_file (str): path to the audio file
    # Output: None
    # Side effect: creates a folder vad_chunks and saves the chunks in it
    
    # Create the vad_chunks folder if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    try:
        # Load the audio file
        audio = pydub.AudioSegment.from_wav(audio_file)
    except:
        print(f'Failed to load {audio_file}')
        return None
    # split audio into 30 second chunks and get start and end time of each chunk
    video_id = audio_file.split('/')[-1].split('.')[0]
    start_time = 0
    chunk_times =[]
    for i, chunk in enumerate(audio[::30000]):
        end_time = start_time + len(chunk) / 1000
        # Save the chunk in the vad_chunks folder
        file_name = f'{video_id}__{start_time}_{end_time}.wav'
        chunk.export(f'{out_dir}/{file_name}', format='wav')
        chunk_times.append((start_time, end_time, file_name))
        start_time = end_time
    return chunk_times


def clean_keyword(keyword):
    return keyword.strip().strip('-').strip()


def convert_timestamps_to_seconds(timestamps):
    # Iterate through each time stamp, check if it's in format H#M#S# or M#S# and convert it to seconds
    times = [t['start'] for t in timestamps]
    keywords = [clean_keyword(t['text']) for t in timestamps]
    for i, timestamp in enumerate(times):
        timestamp = timestamp.split(":")
        # add the time to the timestamp
        times[i] = sum([int(t)*(60**i) for i, t in enumerate(timestamp[::-1])])

    return times, keywords


def check_overlap(interval1, interval2):
    # check if two intervals overlap strictly
    # interval1 and interval2 are tuples of (start, end)
    return interval1[0] < interval2[1] and interval2[0] < interval1[1]


def make_labels(timestamps, keywords, start, end, video_length):
    """"
    timestamps: list of timestamps in seconds
    keywords: list of keywords
    start: start time of the chunk
    end: end time of the chunk
    video_length: length of the video in seconds
    returns: a string of keywords with timestamps in the chunk converted to an interval between 0 and 30 seconds
    """
    label = []
    # find all timestamps that start after start
    if timestamps[0] != 0:
        print(timestamps[0])
    if timestamps[0] != 0:
        timestamps.insert(0, 0.0)
        keywords.insert(0, "Introduction")
    
    if timestamps[-1] != video_length:
        timestamps.append(video_length)
    

    indices = []
    # get all indices of timestamps that are between start and end
    intervals = [(timestamps[i], timestamps[i+1]) for i in range(len(timestamps)-1)]
    query_interval = (start, end)
    # find all intervals that overlap with query interval
    for i, interval in enumerate(intervals):
        if check_overlap(interval, query_interval):
            indices.append(i)

    label_times = [(max(0.0,intervals[i][0]-start), min(30.0, intervals[i][1]-start)) for i in indices]
    label = [keywords[i] for i in indices]
    # format label with <|start_time|>keyword<|end_time|>
    label = ''.join([f"<|{label_times[i][0]}|>{label[i]}<|{label_times[i][1]}|>" for i in range(len(label))])
    return label
    

def data_processing(root_dir, json_file, out_dir, csv_file, data_split='train/'):
    """"
    root_dir: root directory of the dataset
    json_file: path to the json file with the timestamps
    out_dir: path to the output directory
    csv_file: path to the csv file to save the metadata
    data_split: train or test
    returns: None
    side effect: saves the metadata in a csv file
    """

    with open(json_file, 'r') as f:
        timestamps_list = json.load(f)
    filenames = []
    labels = []
    titles = []
    # go through each in timestamps_list and add video_id as a key
    # check if out_dir exists, if not create it

    if os.path.exists(out_dir):
        os.system(f'rm -r {out_dir}')
    
    os.makedirs(out_dir)
    pd.DataFrame({'file_name':[], 'transcription':[], 'title':[]}).to_csv(csv_file, index=None)
    
    for i, audio_file in enumerate(tqdm(os.listdir(root_dir))):
        if audio_file.endswith(".wav") and "temp" not in audio_file and "part" not in audio_file:
            # find url filewhere audio_file is the video id for youtube
            url = 'https://www.youtube.com/watch?v=' + audio_file.split('.wav')[0]
            info = timestamps_list[url]
            title = info['title']
            audio_path = root_dir + f"/{audio_file}"
            chunk_times = split_30_s(audio_path, out_dir)

            if chunk_times is not None:
                title = info['title']
                timestamps = info['timestamps']
                times, keywords = convert_timestamps_to_seconds(timestamps)
                for chunk_start, chunk_end, filename in chunk_times:
                    label = make_labels(times, keywords, chunk_start, chunk_end, video_length=chunk_times[-1][1])
                    labels.append(label)
                    filenames.append(data_split + filename)
                    titles.append(title)
    
            df = pd.DataFrame({'file_name': filenames, 'transcription': labels, 'title' : titles})
            if i% 30 == 0 or i == (len(os.listdir(root_dir))-1):
                print('Writing to csv')
                df.to_csv(csv_file, mode='a', index=False, header=False)
                print('Done writing')
                filenames = []
                labels = []
                titles = []
            


                    
    

def make_splits():
    all_files = [file for file in os.listdir('audio_data') if 'wav' in file]
    all_files = [file for file in all_files if 'temp' not in file and 'part' not in file]
    
    import shutil
    from tqdm import tqdm
    train_dir = 'audio_data/train'
    val_dir = 'audio_data/val'
    # check if train_dir and val_dir exist, if not create them
    if os.path.exists(train_dir):
        # remove it
        os.system(f'rm -r {train_dir}')
    
    os.makedirs(train_dir)
    if os.path.exists(val_dir):
        os.system(f'rm -r {val_dir}')
        print('here')
    
    os.makedirs(val_dir)
    
    # split all files into train and val and move the files into the respective folders, use train_test_spliut
    from sklearn.model_selection import train_test_split
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
    
    for file in tqdm(train_files):
        shutil.copy(f'audio_data/{file}', f'{train_dir}/{file}')
    
    import sys
    sys.exit()
    for file in val_files:
        print('here')
        print(f'{val_dir}/{file}')
        shutil.copy(f'audio_data/{file}', f'{val_dir}/{file}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/thedrive/audio_data')
    parser.add_argument('--out_dir', type=str, default='dataset/train')
    parser.add_argument('--csv_file', type=str, default='dataset/metadata.csv')
    parser.add_argument('--json_file', type=str, default='timestamps_list.json')
    parser.add_argument('--data_split', type=str, default='train/')
    args = parser.parse_args()
    data_processing(args.root_dir, args.json_file, args.out_dir, args.csv_file, data_split=args.data_split)


