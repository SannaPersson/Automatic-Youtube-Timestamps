import os
from datasets import load_dataset, load_from_disk, Audio, DatasetDict, IterableDataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig, PretrainedConfig
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import torch
from torch.utils.data import DataLoader
import datetime
from whisper import tokenizer
import whisper
import re
import gc
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union


metric = evaluate.load("wer")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device="cuda"
MODEL_NAME = "openai/whisper-small"
sampling_rate = 16_000
woptions = whisper.DecodingOptions(language="en", without_timestamps=False)
whisper_tokenizer = tokenizer.get_tokenizer(multilingual=True, language="en", task=woptions.task)

config = PretrainedConfig(name_or_path=MODEL_NAME, use_cache=False, forced_bos_token_id=whisper_tokenizer.sot, forced_eos_token_id=whisper_tokenizer.eot, bos_token_id=whisper_tokenizer.sot, eos_token_id=whisper_tokenizer.eot)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
hf_tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="english", task="transcribe", predict_timestamps=True)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False, use_cache=False).to(device)
model.config.decoder_start_token_id = whisper_tokenizer.sot_prev

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio_arrays = [ex['array'] for ex in batch["audio"]]
    #sampling_rate = audio["sampling_rate"]
    #assert sampling_rate == 16_000, f"Expected sampling rate of 16kHz, got {sampling_rate}Hz instead."

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio_arrays, sampling_rate=sampling_rate).input_features
    transcriptions = batch['transcription']
    batch["labels"] = []
#    batch['decoder_input_ids'] = []
    for i in range(len(transcriptions)):
        # the regular expression below is used to extract timestamps from the transcript

        pattern = re.compile(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>")


        #prev_keywords = str(batch['prev_keywords'][i] or '')
        #prev_keywords = ', '.join(prev_keywords.split(', ')[-5:])
        #prompt = f"Title: {batch['title'][i]}. Previous keywords: {prev_keywords}. Next: "
        prompt = f"Title: {batch['title'][i]}."
        label = [whisper_tokenizer.sot_prev]+ whisper_tokenizer.encode(prompt) + [*whisper_tokenizer.sot_sequence]

        #label = [*whisper_tokenizer.sot_sequence]

        matches = pattern.finditer(transcriptions[i])
        for match in matches:
            start = float(match.group(1))
            text = match.group(2)
            end = float(match.group(3))
            #print(f'Match: {start} {text} {end}')

            start_time_token = int(start/0.02 + whisper_tokenizer.timestamp_begin)
            end_time_token = int(end/0.02 + whisper_tokenizer.timestamp_begin)
            label += [start_time_token] + whisper_tokenizer.encode(text) + [end_time_token]
            

        label += [whisper_tokenizer.eot]
        batch["labels"].append(label)
        #batch["decoder_input_ids"].append([whisper_tokenizer.sot_prev] + label)
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __call__(self, features):
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # pad the labels to max length
        labels_batch = hf_tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == whisper_tokenizer.sot_prev).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch



def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = hf_tokenizer.pad_token_id
    
    # we do not want to group tokens when computing the metrics
    pred_str = hf_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = hf_tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    data = []
    tot = ""
    for pred, label in zip(pred_str, label_str):
        tot += "----------- PREDICTIONS -----------\n"
        tot += pred
        tot += "\n----------- LABELS -----------\n"
        tot += label
        tot += "\n----------- END -----------\n\n"
    data.append(tot)
    # write to file using current time

    now = datetime.datetime.now()
    with open("results/" + now.strftime("%Y-%m-%d-%H-%M-%S") + ".txt", "w") as f:
        f.write("\n\n\n".join(data))

    # make pred and label str into lower case
    pred_str = [pred.lower() for pred in pred_str]
    label_str = [label.lower() for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    model.config.eos_token_id = whisper_tokenizer.eot
    model.config.bos_token_id = whisper_tokenizer.sot


    oscar = load_dataset("audiofolder", data_dir="dataset")["train"]#, cache_dir="/mnt/thedrive/cache")
    print("shuffling")
    oscar = oscar.shuffle(seed=45)
    print(oscar)
    # to get the best speed we don't shuffle the dataset before sharding, and we load shards of contiguous data
    num_shards = 100
    print("Doing sharding")
    shards = [oscar.shard(num_shards=num_shards, index=index, contiguous=True) for index in range(num_shards)]

    def gen_from_shards(shards):
        for shard in shards:
            for example in shard:
                yield example

    
    print("Iterable")
    dataset = IterableDataset.from_generator(gen_from_shards, gen_kwargs={"shards": shards})
    dataset = dataset.shuffle(buffer_size=50000, seed=45)

    print("Mapping")
    dataset = dataset.map(prepare_dataset, batch_size=256, batched=True)#batch_size=128, writer_batch_size=128, num_proc=6, batched=True)

    print("Removing columns")
    dataset = dataset.remove_columns(["audio", "transcription"])#, "title", "prev_keywords"])

    print("Starting")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding()
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=16,
        output_dir="./whisper-small",
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        warmup_steps=50,
        max_steps = 100000,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=500,
        generation_max_length=225,
        logging_steps=5,
        report_to="tensorboard",
        weight_decay=0.1,
        dataloader_num_workers=2,
        ignore_data_skip=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=feature_extractor,
    )

    trainer.train(resume_from_checkpoint=True)

if __name__ == '__main__':
    main()
