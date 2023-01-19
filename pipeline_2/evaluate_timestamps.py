import torch
import argparse
import torchaudio
import whisper
import pandas as pd
import re
import evaluate
from yt_dlp import YoutubeDL

metric = evaluate.load("wer")

# SET Visible cuda environment to "1"
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_what_you_can(pretrained, model, load_running_averages=True, verbose=True):
    """Load a pretrained model into a model, ignoring keys that don't match."""
    is_dict = isinstance(pretrained, dict)
    dict_pretrained = pretrained if is_dict else pretrained.state_dict()
    dict_not_pretrained = model.state_dict()
    iter_dict = dict_not_pretrained.copy()

    for name, param in pretrained.items():
        if "running" in name and not load_running_averages:
            print(f"Skipping running average metric of layer: {name}")
            continue

        if verbose:
            print(f"Current: {name}")
        same_name, same_size = None, None

        for name2, param2 in iter_dict.items():
            if name == name2 and same_name is None:
                same_name = name

            if param.size() == param2.size() and same_size is None:
                same_size = name2

        if (
            same_name
            and dict_not_pretrained[same_name].size()
            == dict_not_pretrained[same_name].size()
        ):
            dict_not_pretrained[same_name].data.copy_(pretrained[same_name])
            if verbose:
                print(f"Successfully loaded (same name): {same_name}")
            del iter_dict[same_name]

        else:
            raise Exception(f"Could not find a matching layer for {name}")

    model.load_state_dict(dict_not_pretrained)
    print("Done")


def load_finetuned_whisper(checkpoint_file, model):
    """Remaps HuggingFace whisper weights to Whisper release weights."""
    pretrained = torch.load(checkpoint_file, map_location="cpu")
    new_key_mappings = {}

    for (hf_key, hf_val), (whisper_key, whisper_val) in zip(
        pretrained.items(), model.state_dict().items()
    ):
        new_hf_key = (
            hf_key.replace("encoder_attn", "cross_attn")
            .replace("model.", "")
            .replace("k_proj", "key")
            .replace("v_proj", "value")
            .replace("q_proj", "query")
            .replace("layers", "blocks")
            .replace("self_attn", "attn")
            .replace("out_proj", "out")
            .replace("fc1", "mlp.0")
            .replace("fc2", "mlp.2")
            .replace("final_layer_norm", "mlp_ln")
            .replace("embed_positions.weight", "positional_embedding")
            .replace("attn_layer_norm", "attn_ln")
            .replace("embed_tokens", "token_embedding")
            .replace("decoder.layer_norm", "decoder.ln")
            .replace("encoder.layer_norm", "encoder.ln_post")
        )
        new_key_mappings[hf_key] = new_hf_key

    new_dict = {}
    for key, val in pretrained.items():
        new_dict[new_key_mappings[key]] = val

    load_what_you_can(new_dict, model, verbose=False)

def download_audio(url, output_file="ex_1.wav"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '128',
        }],
        'writesubtitles': True,
        'writeautomaticsub': False,
        #'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'postprocessor_args': [
            '-ar', '16000',
            '-ac', '1',
            #'-acodec', 'pcm_s16le',
            #'-af', 'loudnorm=I=-14:LRA=11:TP=-1.5',
        ],
        'quiet': True,
        'no_warnings': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])   


def main(model_name, url, checkpoint, output_file="ex_1.wav", prompt=""):
    model = whisper.load_model(model_name).to("cuda")
    checkpoint = f"{checkpoint}/pytorch_model.bin"
    load_finetuned_whisper(checkpoint, model)
    model.eval()

    if os.path.exists(output_file):
        os.remove(output_file)  

    download_audio(url)

    result = model.transcribe(
            output_file, language="en", temperature=0.4, beam_size=10, condition_on_previous_text=False, verbose=True, prompt=prompt)
    )



if __name__ == "__main__":
    # get arguments for model name, url, checkpoint, output file, and prompt with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="small", help="Model name")
    parser.add_argument("--url", type=str, help("URL of video to transcribe"))
    parser.add_argument("--checkpoint", type=str, default="whisper-small", help="Path to checkpoint folder which contains pytorch_model.bin")
    parser.add_argument("--output_file", type=str, default="ex_1.wav", help="Path to output file")
    parser.add_argument("--prompt", type=str, default="", help="Prompt to use for transcription")

    main(args.model_name, args.url, args.checkpoint, args.output_file, args.prompt)
