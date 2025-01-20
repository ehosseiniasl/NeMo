import argparse
import csv
import json
import os
import shutil
from io import BytesIO
from pathlib import Path

### from nemo.collections.tts.models import AudioCodecModel
import librosa
import numpy as np
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

from nemo.utils import logging
import re
import ipdb

#  python -m pdb -c continue /lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/create_shars_duplex_multi_from_single.py --manifest /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.conversation_style_manifest_normalized_with_correctpath_with_evaluations.json.200 --out_shar_dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.b.duplex.200/shars --num_shard 1


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

def get_segments(input_string):
    try:
        timestamps = re.findall(r"<\|(\d+)\|>", input_string)
        words = re.findall(r">([^<]+)<", input_string)
        
        if len(timestamps) == 0 or len(timestamps) % 2 != 0:
            # ipdb.set_trace()
            return []

        # Convert time segments to integers for easier calculations
        # time_segments = [(int(start), int(end)) for start, end in time_segments]
        time_segments = [(int(timestamps[i]), int(timestamps[i + 1])) for i in range(0, len(timestamps), 2)]
        if len(words) != len(time_segments):
            return []

        # Initialize the result lists
        grouped_segments = []
        current_segment = []
        current_words = []
        # try:
        current_start = time_segments[0][0]
        # except:
        #     ipdb.set_trace()

        # Iterate through the time segments
        for i in range(len(time_segments)):
            if i > 0:
                # Calculate the gap between the end of the previous segment and the start of the current segment
                previous_end = time_segments[i - 1][1]
                current_start_segment = time_segments[i][0]
                if current_start_segment - previous_end > 12:
                    # If gap > 12, finalize the current group
                    current_end = time_segments[i - 1][1]
                    grouped_segments.append({
                        'start': current_start,
                        'end': current_end,
                        'words': current_words
                    })
                    # Start a new group
                    current_segment = []
                    current_words = []
                    current_start = current_start_segment

            # Add the current segment and word to the group
            # try:
            current_segment.append(time_segments[i])
            current_words.append(words[i])
            # except:
            #     ipdb.set_trace()

        # Append the last group
        if current_segment:
            current_end = time_segments[-1][1]
            grouped_segments.append({
                'start': current_start,
                'end': current_end,
                'words': current_words
            })

        # # Print the results
        # print("Grouped segments:")
        # for segment in grouped_segments:
        #     print(segment)
        return grouped_segments
    except:
        return []


def create_shar_from_manifest(manifest, out_shar_dir, num_shard=10, dataset_name='squadv2', turn_silence_sec=0.32):
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")
    shard_size = int(len(in_manifest) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    user_recordings = []
    answer_list = []
    instructions = []
    stereo_recordings = []
    source_language = []
    target_language = []
    target_recordings = []
    # cleaned_manifest = []
    for i, line in tqdm(enumerate(in_manifest)):
        # For single turn convs is a list of 2 elements
        # First element is user speech and second is agent speech
        # convs = line["conversations"]
        # for conv in convs:
        #     conv["value"] = conv["value"].replace("fs7", "fsw")

        # if convs[1]["transcript"] != 'I could not find the answer in the audio.':
        #     try:
        # if 'value' not in convs[0] or not os.path.exists(convs[0]['value']):
        #     raise FileNotFoundError(f"File not found for convs[0]['value']: {convs[0]['value']}")

        # if 'value' not in convs[1] or not os.path.exists(convs[1]['value']):
        #     raise FileNotFoundError(f"File not found for convs[1]['value']: {convs[1]['value']}")

        # cleaned_manifest.append(line)
        # User_Speech
        # user_recording = Recording.from_file(convs[0]['value'])
        # user_recordings.append(user_recording)
        stereo_recording = Recording.from_file(line['audio_filepath'])
        stereo_recordings.append(stereo_recording)

        user_recording = Recording.from_file(line['user_audio_filepath'])
        user_recordings.append(user_recording)

        # Instructions from the user. In case the question is part of the source audio this is a static text "Transcribe and answer",
        # If not then this is the actual question from the user but in text.
        # For direct_s2s instructions are always empty (else part)
        # if "instruction" in convs[0]:
        #     instructions.append(convs[0]["instruction"])
        # else:
        #     instructions.append("")

        # Language source
        # if "lang" in convs[0]:
        #     source_language.append(convs[0]["lang"])
        # else:
        #     source_language.append("EN")
        source_language.append(line['source_lang'])

        # Loading agent audio and using only the extracted features as nd.array
        # target_recordings.append(Recording.from_file(convs[1]['value']))
        target_recordings.append(Recording.from_file(line['assistant_audio_filepath']))

        # # Agent answer transcript
        # answer_list.append(convs[1]["transcript"])
        # # Language target
        # if "lang" in convs[1]:
        #     target_language.append(convs[1]["lang"])
        # else:
        #     target_language.append("EN")
    # except:
    #     logging.info(f'Skipping {i}th json record.')
    # in_manifest = cleaned_manifest

    print("Done extracting data from manifest")
    # print(len(user_recordings))
    # cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))
    print(len(stereo_recordings))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(stereo_recordings))
    framerate = 80 / 1000
    # Attach text
    num_cuts = len(cuts)
    for j, cut in tqdm(enumerate(cuts)):
        user_audio_list = []
        agent_audio_list = []
        total_dur = 0

        
        # convs = in_manifest[j]["conversations"] + in_manifest[num_cuts - j - 1]["conversations"]
        # for i in range(0, len(convs), 2):
        convs = in_manifest[j]
        
        # user_recording = Recording.from_file(convs[i]['value'])
        # agent_recording = Recording.from_file(convs[i + 1]['value'])
        user_recording = user_recordings[j]
        agent_recording = target_recordings[j]

        sample_rate = agent_recording.sampling_rate
        # user_duration = user_recording.duration + turn_silence_sec
        user_duration = user_recording.duration
        agent_duration = agent_recording.duration
        # cur_user_audio = user_recording.resample(sample_rate).load_audio()
        # cur_agent_audio = agent_recording.load_audio()

        # silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
        # user_audio_list.extend([cur_user_audio, silence_padding, np.zeros_like(cur_agent_audio)])
        # agent_audio_list.extend([np.zeros_like(cur_user_audio), silence_padding, cur_agent_audio])
        user_segments = get_segments(in_manifest[j]['user_pred_text'])
        if len(user_segments) == 0 :
            continue
        for segment in user_segments:
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=segment['start'] * framerate, #total_dur,
                    duration=(segment['end']-segment['start']) * framerate, #user_duration,
                    text="".join(segment['words']).strip(), #convs[i]["instruction"],
                    speaker="User", #convs[i]["from"],
                    language="EN",
                ),
            )

        agent_segments = get_segments(in_manifest[j]['assistant_pred_text'])
        if len(agent_segments) == 0:
            continue
        for segment in agent_segments:
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=segment['start'] * framerate, #total_dur,
                    duration=(segment['end']-segment['start']) * framerate, #user_duration,
                    text="".join(segment['words']).strip(), #convs[i]["instruction"],
                    speaker="Assistant", #convs[i]["from"],
                    language="EN",
                ),
            )

        # cut.supervisions.append(
        #     SupervisionSegment(
        #         id=cut.id,
        #         recording_id=cut.id,
        #         start=total_dur,
        #         duration=user_duration,
        #         text=convs[i]["instruction"],
        #         speaker=convs[i]["from"],
        #         language="EN",
        #     ),
        # )
        # cut.supervisions.append(
        #     SupervisionSegment(
        #         id=cut.id,
        #         recording_id=cut.id,
        #         start=total_dur + user_duration,
        #         duration=agent_duration,
        #         text=convs[i + 1]["transcript"],
        #         speaker=convs[i + 1]["from"],
        #         language="EN",
        #     ),
        # )
        # total_dur += user_duration + agent_duration
        # total_dur = user_duration

        # user_audio = np.concatenate(user_audio_list, axis=1)
        # agent_audio = np.concatenate(agent_audio_list, axis=1)
        # append trailing silence to help agent learn to stop
        # user_audio_list.append(silence_padding)
        # agent_audio_list.append(silence_padding)

        # cut.duration = total_dur + turn_silence_sec
        # cut.duration_no_sil = total_dur
        cut.duration = in_manifest[j]['duration']
        # cut.start = 0.0

        # user_stream = BytesIO()
        # agent_stream = BytesIO()
        # save_audio(dest=user_stream, src=user_audio, sampling_rate=sample_rate, format="wav")
        # save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sample_rate, format="wav")
        # user_stream.seek(0)
        # agent_stream.seek(0)
        # cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut.id}_user")
        # cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut.id}_agent")
        
        # cut.source_audios.append(user_recording)
        # cut.target_audios.append(agent_recording)
        cut.recording = user_recording
        cut.target_audio = agent_recording


    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    shard_size = shard_size
    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(
        out_shar_dir, fields={"recording": "wav", "target_audio": "wav"}, num_jobs=1, shard_size=shard_size
    )
    print(f"...share created")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        type=str,
        default="/lustre/fsw/portfolios/convai/users/subhankarg/manifests/s2s/squadv2/conversation_style_manifest_normalized_with_correctpath_with_evaluations.json",
    )
    parser.add_argument(
        '--out_shar_dir',
        type=str,
        default="/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/s2s_lhotse_with_wavs/squadv2/",
    )
    parser.add_argument(
        '--num_shard',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="squadv2",
    )

    args = parser.parse_args()
    print(f"manifest {args.manifest}")
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
