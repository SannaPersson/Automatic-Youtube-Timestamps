"""
A YouTube description often contains timestamps. We want to extract these timestamps and the corresponding text.
The input is a long string containing the description of a YouTube video with a lot of text.

Example outputs of this function:
0:00 - Introduction
0:30 - What is the problem?
1:00 - What is the solution?

But the timestamps could also be in the format:
00:00 - Introduction
00:30 - What is the problem?
01:00 - What is the solution?

Or even:
00:00:00 - Introduction
00:00:30 - What is the problem?
00:01:00 - What is the solution?

We want to extract the timestamps and the chapter titles.
Complete the function with huggingface's transformers library.
"""


import re
import pandas as pd
import os
import json


def get_timestamps(description):
    timestamps = []
    timestamp_regex = r"\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2}"
    timestamp_matches = re.findall(timestamp_regex, description)
    for timestamp in timestamp_matches:
        timestamps.append(timestamp)
    return timestamps


def process_chapter(chapter):
    # strip chapter from whitespaces in beginning and end of string
    chapter = chapter.strip()
    chapter = chapter.strip("-")

    return chapter


def get_text(description, timestamps):
    """
    Extract description text for each timestamp.
    """
    text = []
    for i, timestamp in enumerate(timestamps):
        chapter = description.split(timestamp)[1].split("\n")[0]
        text.append(process_chapter(chapter))

    return timestamps, text


def main():
    # Loop through all descriptions in youtube_data.csv and extract timestamps print them
    # open df youtube_data.csv
    # loop through all descriptions
    # get timestamps
    # print timestamps
    transcript_path = "transcripts_v2"
    df = pd.read_csv("youtube_data_v2.csv")
    num_labels = 0
    for description in df["description"]:
        # get the video link from df
        try:
            video_link = df[df["description"] == description]["url"].values[0]
            timestamps = get_timestamps(description)
        except:
            print("Probably didn't have a description")
            continue

        # convert video link to video id
        video_id = video_link.split("=")[1]
        # check if video id in transcript_path
        if video_id + ".json" in os.listdir(transcript_path):
            num_labels += 1
        else:
            continue

        timestamps_lbl = ""
        if len(timestamps) > 0:
            print(f"Video link: {video_id}")
            timestamps, chapters = get_text(description, timestamps)

            # Go through timestamps, chapters and print them
            for i, timestamp in enumerate(timestamps):
                timestamps_lbl += f"{timestamp} - {chapters[i]}\n"

            # find json in transcripts folder that matches video id and open it
            with open(f"{transcript_path}/{video_id}.json", "r") as f:
                transcript = json.load(f)
                # if data is a list, then make it a dictionary with key "segments" and value the list
                if type(transcript) == list:
                    transcript = {"segments": transcript}

            # add labels to transcript
            transcript["timestamps"] = timestamps_lbl

            print(transcript["timestamps"])

            # close json file
            with open(f"{transcript_path}/{video_id}.json", "w") as f:
                json.dump(transcript, f)

    print(f"Number of labels: {num_labels}")


if __name__ == "__main__":
    main()
