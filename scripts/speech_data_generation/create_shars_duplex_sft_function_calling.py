import argparse
import csv
import json
import os
import shutil
from pathlib import Path

### from nemo.collections.tts.models import AudioCodecModel
import librosa
import numpy as np
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm
import ipdb

def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(manifest, out_shar_dir, audio_dir, num_shard=10):
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")
    shard_size = int(len(in_manifest) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    # user_recordings = []
    conv_recordings = []
    answer_list = []
    instructions = []
    source_language = []
    target_language = []
    target_recordings = []
    audio_dir = "/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/digital_human_alm/data/SFT/ameya_data/synthesized/audio"
    for i, line in tqdm(enumerate(in_manifest)):
        # For single turn convs is a list of 2 elements
        # First element is user speech and second is agent speech
        convs = line["conversations"]
        for conv in convs:
            conv["value"] = conv["value"].replace("fs7", "fsw")

        # User_Speech
        # user_recording = Recording.from_file(convs[0]['value'])
        # user_recordings.append(user_recording)
        found_image = False
        for turn in convs:
            if 'audio_value' in turn:
                conv_recording = Recording.from_file(os.path.join(audio_dir, turn['audio_value']))
                conv_recordings.append(conv_recording)
                found_image = True
                break
        assert found_image == True
        # Instructions from the user. In case the question is part of the source audio this is a static text "Transcribe and answer",
        # If not then this is the actual question from the user but in text.
        # For direct_s2s instructions are always empty (else part)
        # if "instruction" in convs[0]:
        #     instructions.append(convs[0]["instruction"])
        # else:
        #     instructions.append("")
        instructions.append(line["system"])

        # Language source
        if "lang" in convs[0]:
            source_language.append(convs[0]["lang"])
        else:
            source_language.append("EN")

        # Loading agent audio and using only the extracted features as nd.array
        # target_recordings.append(Recording.from_file(convs[1]['value']))
        # Agent answer transcript
        # answer_list.append(convs[1]["transcript"])
        # Language target
        if "lang" in convs[1]:
            target_language.append(convs[1]["lang"])
        else:
            target_language.append("EN")

    print("Done extracting data from manifest")
    # print(len(user_recordings))
    print(len(conv_recordings))
    # cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(conv_recordings))

    # Attach text
    for i, cut in tqdm(enumerate(cuts)):
        convs = in_manifest[i]["conversations"]
        cut.target_audios = []
        cut.source_audios = []
        total_duration = 0
        cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=0,
                    duration=0, #cut.recording.duration,
                    text=instructions[i],
                    speaker="system", 
                    language="EN",
                ),
            )
        for i in range(0, len(convs), 2):
            if 'audio_value' not in convs[i]:
                user_function = convs[i]['value']
                user_transcript = ""
                user_audio = "" #np.zeros((0,0))
                user_duration = 0
            else:
                user_function = ""
                user_transcript = convs[i]['value_normalized']
                user_audio = os.path.join(audio_dir, convs[i]['audio_value'])
                user_duration = Recording.from_file(user_audio).duration
            # ipdb.set_trace()
            
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=total_duration, #0,
                    duration=user_duration, #cut.recording.duration,
                    # text=convs[i]["instruction"],
                    text=user_transcript,
                    custom={'function': user_function},
                    speaker=convs[i]["from"],
                    language="EN",
                ),
            )

            total_duration += user_duration 

            if 'audio_value' not in convs[i+1]:
                assistant_function = convs[i+1]['value']
                assistant_transcript = ""
                assistant_audio = "" #np.zeros((0,0))
                assistant_duration = 0
            else:
                assert 'audio_value' in convs[i+1]
                assistant_function = ""
                assistant_transcript = convs[i+1]['value_normalized']
                assistant_audio = os.path.join(audio_dir, convs[i+1]['audio_value'])
                assistant_duration = Recording.from_file(assistant_audio).duration

            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=total_duration,
                    duration=assistant_duration, #cut.recording.duration,
                    # text=convs[i + 1]["transcript"],
                    text=assistant_transcript,
                    custom={'function': assistant_function},
                    speaker=convs[i + 1]["from"],
                    language="EN",
                ),
            )
            total_duration += assistant_duration
            # cut.source_audios.append(Recording.from_file(convs[i]['value']))
            # cut.target_audios.append(Recording.from_file(convs[i + 1]['value']))
            if user_audio != "":
                cut.source_audios.append(Recording.from_file(user_audio))
            else:
                # cut.source_audios.append(np.zeros((1,1)))
                cut.source_audios.append(Recording(sources=[], id='', sampling_rate=0, num_samples=1, duration=0))
            
            if assistant_audio != "":
                # ipdb.set_trace()
                cut.target_audios.append(Recording.from_file(assistant_audio))
            else:
                # cut.target_audios.append(np.zeros((1,1)))
                cut.target_audios.append(Recording(sources=[], id='', sampling_rate=0, num_samples=1, duration=0))
        cut.duration = total_duration

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    shard_size = shard_size
    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(out_shar_dir, fields={"source_audio": "wav", "target_audio": "wav"}, num_jobs=4, shard_size=shard_size)
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
        '--audio_dir',
        type=str,
        default="/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/s2s_lhotse_with_wavs/squadv2/",
    )
    parser.add_argument(
        '--num_shard',
        type=int,
        default=10,
    )

    args = parser.parse_args()
    print(f"manifest {args.manifest}")
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"out_shar_dir {args.audio_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        audio_dir=args.audio_dir,
        num_shard=args.num_shard,
    )


if __name__ == "__main__":
    main()
