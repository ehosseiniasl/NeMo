import argparse
import csv
import json
import os
import shutil
from pathlib import Path

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
import ipdb
from io import BytesIO
import tempfile

os.environ["LHOTSE_STRICT"] = "0"

def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

def num_function_turns(data):
    c = 0
    for i, d in enumerate(data):
        if i % 2 == 0 and ("<TOOLCALL>" in d['value'] or "<TOOL_RESPONSE>" in d['value']):
            c += 1
        elif i % 2 != 0 and ("<TOOLCALL>" in d['value'] or "<TOOL_RESPONSE>" in d['value']):
            c += 1
    return c

def create_shar_from_manifest(manifest, out_shar_dir, audio_dir, num_shard=10, overlap_sec=0.64):
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")
    
    # Debug print for first manifest entry
    if len(in_manifest) > 0:
        print("\nDebug - First manifest entry structure:")
        print(json.dumps(in_manifest[0], indent=2))
        print("\nDebug - First conversation structure:")
        print(json.dumps(in_manifest[0]["conversations"][0], indent=2))
        if "original_manifest" in in_manifest[0]["conversations"][0]:
            print("\nDebug - Original manifest structure:")
            print(json.dumps(in_manifest[0]["conversations"][0]["original_manifest"], indent=2))
            if "raw_data" in in_manifest[0]["conversations"][0]["original_manifest"]:
                print("\nDebug - Raw data structure:")
                print(json.dumps(in_manifest[0]["conversations"][0]["original_manifest"]["raw_data"], indent=2))
    
    shard_size = int(len(in_manifest) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    conv_recordings = []
    answer_list = []
    instructions = []
    source_language = []
    target_language = []
    target_recordings = []
    valid_manifest = []
    
    # First pass: validate audio files
    for i, line in tqdm(enumerate(in_manifest)):
        try:
            user_path = os.path.join(audio_dir, line["conversations"][0]['audio_value'].split("/")[-1])
            user_audio = Recording.from_file(user_path).load_audio()
            if user_audio is not None:
                valid_manifest.append(line)
        except:
            print(f"crashed audio: {user_path}")
            continue

    print(f"num manifests: {len(in_manifest)}")
    print(f"num valid manifests: {len(valid_manifest)}")

    for i, line in tqdm(enumerate(valid_manifest)):
        convs = line["conversations"]
        conv_recording = Recording.from_file(os.path.join(audio_dir, convs[0]['audio_value'].split("/")[-1]))
        sample_rate = conv_recording.sampling_rate
        conv_recordings.append(conv_recording)

        function = line["conversations"][0]["original_manifest"]["raw_data"]["function"]
        system_prompt = f"<AVAILABLE_TOOLS>{function}</AVAILABLE_TOOLS>"
        instructions.append(system_prompt)
        source_language.append("EN")
        target_language.append("EN")

    print("Done extracting data from manifest")
    print(len(conv_recordings))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(conv_recordings))

    unequal = 0
    for j, cut in tqdm(enumerate(cuts)):
        turn_silence_sec = 0 #0.32
        silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
        overlap_samples = int(overlap_sec*sample_rate)

        user_audio = np.array([[]])
        agent_audio = np.array([[]])
        total_dur = 0

        convs = valid_manifest[j]["conversations"]
        total_duration = 0
        
        # Add system prompt supervision
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=0,
                text=instructions[j],
                speaker="system", 
                language="EN",
            ),
        )

        # Process user audio
        user_function = ""
        user_transcript = convs[0]['normalized_text_value']
        user_path = os.path.join(audio_dir, convs[0]['audio_value'].split("/")[-1])
        user_duration = Recording.from_file(user_path).duration
        
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=total_duration,
                duration=user_duration,
                text=user_transcript,
                custom={'function': user_function},
                speaker=convs[0]["from"],
                language="EN",
            ),
        )

        sample_rate = Recording.from_file(user_path).sampling_rate
        try:
            cur_user_audio = Recording.from_file(user_path).resample(sample_rate).load_audio()
        except:
            ipdb.set_trace()
        
        if user_duration > 0:
            total_duration += (user_duration + turn_silence_sec)

        # Process assistant audio
        print(f"\nDebug - Processing cut {cut.id}")
        print(f"Debug - Full manifest entry keys: {list(valid_manifest[j].keys())}")
        print(f"Debug - Conversations structure: {list(valid_manifest[j]['conversations'][0].keys())}")
        print(f"Debug - Original manifest structure: {list(valid_manifest[j]['conversations'][0]['original_manifest'].keys())}")
        print(f"Debug - Raw data keys: {list(valid_manifest[j]['conversations'][0]['original_manifest']['raw_data'].keys())}")
        
        # Try to find ground truth in different possible locations
        ground_truth = None
        if "ground_truth" in valid_manifest[j]:
            ground_truth = valid_manifest[j]["ground_truth"]
        elif "ground_truth" in valid_manifest[j]["conversations"][0]:
            ground_truth = valid_manifest[j]["conversations"][0]["ground_truth"]
        elif "ground_truth" in valid_manifest[j]["conversations"][0]["original_manifest"]:
            ground_truth = valid_manifest[j]["conversations"][0]["original_manifest"]["ground_truth"]
        elif "ground_truth" in valid_manifest[j]["conversations"][0]["original_manifest"]["raw_data"]:
            ground_truth = valid_manifest[j]["conversations"][0]["original_manifest"]["raw_data"]["ground_truth"]
            
        if ground_truth is not None:
            print(f"Debug - Ground truth value: {ground_truth}")
            print(f"Debug - Ground truth type: {type(ground_truth)}")
            
            if isinstance(ground_truth, list) and ground_truth:
                if len(ground_truth) == 1:
                    # For single dictionary, convert to name/arguments format
                    if isinstance(ground_truth[0], dict):
                        for func_name, args in ground_truth[0].items():
                            formatted_call = {
                                "name": func_name,
                                "arguments": args
                            }
                            assistant_function = f"<TOOLCALL>[{formatted_call}]</TOOLCALL>"
                    else:
                        assistant_function = f"<TOOLCALL>[{ground_truth[0]}]</TOOLCALL>"
                else:
                    # For multiple items, format each dictionary
                    formatted_calls = []
                    for call in ground_truth:
                        if isinstance(call, dict):
                            for func_name, args in call.items():
                                formatted_call = {
                                    "name": func_name,
                                    "arguments": args
                                }
                                formatted_calls.append(formatted_call)
                        else:
                            formatted_calls.append(call)
                    assistant_function = f"<TOOLCALL>{formatted_calls}</TOOLCALL>"
            elif isinstance(ground_truth, str) and ground_truth.strip():
                assistant_function = f"<TOOLCALL>[{ground_truth}]</TOOLCALL>"
            else:
                print(f"Debug - Empty or invalid ground truth: {ground_truth}")
                assistant_function = ""
        else:
            print("Debug - No ground_truth field found in any expected location")
            assistant_function = ""
            
        print(f"Debug - Assistant function: {assistant_function}")
        assistant_transcript = ""
        assistant_path = "" 
        assistant_duration = 0

        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=total_duration,
                duration=assistant_duration,
                text=assistant_transcript,
                custom={'function': assistant_function},
                speaker="assistant",
                language="EN",
            ),
        )

        cur_agent_audio = np.zeros_like(cur_user_audio) # assistant is empty, use user duration. 
        assistant_duration = user_duration
        # cur_agent_audio = np.zeros((1, int(assistant_duration * sample_rate)))

        total_duration += assistant_duration        

        # Combine audio
        if cur_user_audio is not None and cur_agent_audio is not None:
            if j != 0 and overlap_samples > 0:
                user_audio = user_audio[:, :-overlap_samples]
                total_dur -= overlap_sec
            
            user_audio = np.concatenate([user_audio, cur_user_audio, silence_padding, 0 * cur_agent_audio], axis=1)
            
            if j != 0 and overlap_samples > 0:
                agent_audio = np.concatenate([agent_audio, 0 * cur_user_audio[:, :-overlap_samples], silence_padding, cur_agent_audio], axis=1)
            else:
                agent_audio = np.concatenate([agent_audio, 0 * cur_user_audio, silence_padding, cur_agent_audio], axis=1)

        # Ensure audio lengths match
        if user_audio.shape != agent_audio.shape:
            unequal += 1
            diff = agent_audio.shape[1] - user_audio.shape[1]
            if diff > 0:
                user_audio = np.concatenate([user_audio, np.zeros((1, int(diff)))], axis=1)
            else:
                agent_audio = np.concatenate([agent_audio, np.zeros((1, int(np.abs(diff))))], axis=1)
            assert user_audio.shape == agent_audio.shape

        # Save user audio
        user_audio_path = f"/tmp/u{j}1.wav"
        save_audio(user_audio_path, user_audio, sample_rate)
        user_recording = Recording.from_file(user_audio_path)
        # Update num_samples to match actual audio data
        user_recording.num_samples = user_audio.shape[1]
        user_recording.duration = user_recording.num_samples / sample_rate
        cut.recording = user_recording

        # Save agent audio
        agent_audio_path = f"/tmp/u{j}2.wav"
        save_audio(agent_audio_path, agent_audio, sample_rate)
        agent_recording = Recording.from_file(agent_audio_path)
        # Update num_samples to match actual audio data
        agent_recording.num_samples = agent_audio.shape[1]
        agent_recording.duration = agent_recording.num_samples / sample_rate
        cut.target_audio = agent_recording

        # Set cut duration and start
        cut.duration = max(user_recording.duration, agent_recording.duration)
        cut.start = 0.0

    print(f"unequal examples: {unequal}")

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out invalid cuts
    valid_cuts_list = []
    for cut in cuts:
        try:
            # Try to load both user and agent audio
            user_audio = cut.recording.load_audio()
            agent_audio = cut.target_audio.load_audio()
            if user_audio is not None and agent_audio is not None:
                valid_cuts_list.append(cut)
        except Exception as e:
            print(f"Error loading cut {cut.id}: {str(e)}")
            continue
    valid_cuts = CutSet.from_cuts(valid_cuts_list)
    
    print(f"num cuts: {len(cuts)}")
    print(f"num valid cuts: {len(valid_cuts)}")

    if len(valid_cuts) == 0:
        print("No valid cuts found. Please check the audio files and paths.")
        return

    # Create shar files
    exported = valid_cuts.to_shar(out_shar_dir, fields={}, num_jobs=1, shard_size=shard_size)
    print(f"...share created")
    
    # Create audio tar files
    for i, path in tqdm(enumerate(exported["cuts"])):
        out_path = path.replace("cuts", "target_audio").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="wav") as writer:
            for cut in CutSet.from_file(path):
                writer.write(
                    cut.id, cut.target_audio.load_audio(), manifest=cut.target_audio, sampling_rate=sample_rate
                )
        out_path = path.replace("cuts", "recording").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="wav") as writer:
            for cut in CutSet.from_file(path):
                writer.write(cut.id, cut.recording.load_audio(), manifest=cut.recording, sampling_rate=sample_rate)

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
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.64,
    )

    args = parser.parse_args()
    print(f"manifest {args.manifest}")
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"out_shar_dir {args.audio_dir}")
    print(f"num_shard {args.num_shard}")
    print(f"overlap {args.overlap}")

    create_shar_from_manifest(
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        audio_dir=args.audio_dir,
        num_shard=args.num_shard,
        overlap_sec=args.overlap,
    )

if __name__ == "__main__":
    main() 