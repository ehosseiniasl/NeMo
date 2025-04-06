#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment, MonoCut
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.utils import Pathlike
from matplotlib import pyplot as plt
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.utils import logging

# Set environment variable for duration mismatch tolerance
os.environ['DURATION_MISMATCH_TOLERANCE'] = '0.1'  # 100ms tolerance

def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

def create_shards(
    manifest_path: Pathlike,
    output_dir: Pathlike,
    audio_dir: Pathlike,
    answer_audio_dir: Pathlike,
    shard_size: int = 1000,
    shard_prefix: str = "cuts",
    num_jobs: int = 1,
    random_seed: int = 42,
    turn_silence_sec: float = 0.32,
    speaker_after_user: str = "assistant",
) -> None:
    """Create shards from a manifest with multiple speakers."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory for audio files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Read manifest
        in_manifest = list(json_reader(manifest_path))
        print(f"Read {len(in_manifest)} entries from manifest")

        # Check if required files exist and clean manifest
        cleaned_manifest = []
        excluded_german = 0
        excluded_missing_files = 0
        
        for i, entry in tqdm(enumerate(in_manifest)):
            try:
                # Skip entries with unwanted question text
                if "Transcribe the spoken content to written German text" in entry.get("question", ""):
                    logging.info(f'Skipping {i}th json record: contains German transcription request')
                    excluded_german += 1
                    continue

                # Get audio paths and construct full paths
                audio_filepath = entry["audio_filepath"]
                audio_wav = os.path.join(audio_dir, audio_filepath)
                question_wav = entry.get("question_wav")  # Make question_wav optional
                target_wav = os.path.join(answer_audio_dir, entry["target_wav"])
                
                # Check if required files exist
                if not os.path.exists(audio_wav):
                    raise FileNotFoundError(f"File not found: {audio_wav}")
                if question_wav and not os.path.exists(question_wav):  # Only check if question_wav is provided
                    raise FileNotFoundError(f"File not found: {question_wav}")
                if not os.path.exists(target_wav):
                    raise FileNotFoundError(f"File not found: {target_wav}")
                    
                cleaned_manifest.append(entry)
                
            except Exception as e:
                logging.info(f'Skipping {i}th json record: {str(e)}')
                excluded_missing_files += 1
                
        in_manifest = cleaned_manifest
        print(f"Number of valid recordings: {len(in_manifest)}")
        print(f"Number of excluded German transcription examples: {excluded_german}")
        print(f"Number of excluded examples due to missing files: {excluded_missing_files}")
        print(f"Total excluded examples: {excluded_german + excluded_missing_files}")

        # Calculate number of cuts based on processing 4 samples at a time
        num_cuts = len(in_manifest) // 4
        print(f"Will create {num_cuts} cuts, processing 4 samples at a time")

        # Create list to hold all cuts
        all_cuts = []
        
        # Create and process all cuts
        for j in tqdm(range(num_cuts)):
            try:
                # Create initial cut
                cut = MonoCut(
                    id=f"cut_{j}",
                    start=0,
                    duration=0,  # Will be updated later
                    channel=0,
                    recording=None,  # Will be set later
                    supervisions=[]
                )
                
                user_audio_list = []
                agent_audio_list = []
                total_dur = 0

                # Take 2 from start and 2 from end
                entries = []
                entries.append(in_manifest[j * 2])                    # First from start
                entries.append(in_manifest[j * 2 + 1])               # Second from start
                entries.append(in_manifest[len(in_manifest) - j * 2 - 2])  # Second from end
                entries.append(in_manifest[len(in_manifest) - j * 2 - 1])  # First from end
                
                # Process each entry individually
                for entry_idx, entry in enumerate(entries):
                    try:
                        # Get audio paths and construct full paths
                        audio_filepath = entry["audio_filepath"]
                        audio_wav = os.path.join(audio_dir, audio_filepath)
                        question_wav = entry.get("question_wav")
                        target_wav = os.path.join(answer_audio_dir, entry["target_wav"])
                        
                        # Load and concatenate both audio files
                        user_audio_data, user_sr = sf.read(audio_wav)
                        if question_wav:
                            question_audio_data, question_sr = sf.read(question_wav)
                            # Ensure same sample rate
                            if question_sr != user_sr:
                                # Resample question audio to match user audio
                                question_audio = Recording.from_file(question_wav)
                                question_audio = question_audio.resample(user_sr)
                                question_audio_data = question_audio.load_audio()
                            # Convert to mono if stereo
                            if len(user_audio_data.shape) > 1:
                                user_audio_data = user_audio_data[:, 0]
                            if len(question_audio_data.shape) > 1:
                                question_audio_data = question_audio_data[:, 0]
                            # Reshape for concatenation
                            user_audio_data = user_audio_data.reshape(1, -1)
                            question_audio_data = question_audio_data.reshape(1, -1)
                            # Concatenate the audio data
                            user_recording_data = np.concatenate([user_audio_data, question_audio_data], axis=1)
                        else:
                            # Convert to mono if stereo
                            if len(user_audio_data.shape) > 1:
                                user_audio_data = user_audio_data[:, 0]
                            # Reshape for concatenation
                            user_recording_data = user_audio_data.reshape(1, -1)

                        # Save concatenated user audio
                        user_temp_path = os.path.join(temp_dir, f'final_user_{j}.wav')
                        sf.write(user_temp_path, user_recording_data.T, user_sr)
                        user_recording = Recording.from_file(user_temp_path)

                        # Load and process agent audio
                        agent_audio_data, agent_sr = sf.read(target_wav)
                        if len(agent_audio_data.shape) > 1:
                            agent_audio_data = agent_audio_data[:, 0]
                        agent_audio_data = agent_audio_data.reshape(1, -1)
                        
                        # Save agent audio
                        agent_temp_path = os.path.join(temp_dir, f'final_agent_{j}.wav')
                        sf.write(agent_temp_path, agent_audio_data.T, agent_sr)
                        agent_recording = Recording.from_file(agent_temp_path)
                        
                        # Create supervision segments
                        user_supervision = SupervisionSegment(
                            id=os.path.basename(audio_wav),
                            recording_id=os.path.basename(audio_wav),
                            start=total_dur,
                            duration=user_recording.duration,
                            text=f"{entry.get('text', '')} {entry.get('question', '')}",
                            speaker="user",
                            language=entry.get("source_lang", "EN"),
                        )
                        
                        agent_supervision = SupervisionSegment(
                            id=os.path.basename(target_wav),
                            recording_id=os.path.basename(target_wav),
                            start=total_dur + user_recording.duration + turn_silence_sec,
                            duration=agent_recording.duration,
                            text=entry.get("answer", ""),
                            speaker=speaker_after_user,
                            language=entry.get("target_lang", "EN"),
                        )

                        # Add supervisions to cut
                        cut.supervisions.append(user_supervision)
                        cut.supervisions.append(agent_supervision)

                        # Process audio
                        sample_rate = agent_recording.sampling_rate
                        user_duration = user_recording.duration + turn_silence_sec
                        agent_duration = agent_recording.duration
                        cur_user_audio = user_recording.load_audio()
                        cur_agent_audio = agent_recording.load_audio()

                        silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
                        user_audio_list.extend([cur_user_audio, silence_padding, np.zeros_like(cur_agent_audio)])
                        agent_audio_list.extend([np.zeros_like(cur_user_audio), silence_padding, cur_agent_audio])

                        total_dur += user_duration + agent_duration

                    except Exception as e:
                        print(f"Error processing entry {entry_idx} in cut {j}: {str(e)}")
                        continue

                # Process final audio
                user_audio = np.concatenate(user_audio_list, axis=1)
                agent_audio = np.concatenate(agent_audio_list, axis=1)

                # Save final audio files
                final_user_path = os.path.join(temp_dir, f'final_user_{j}.wav')
                final_agent_path = os.path.join(temp_dir, f'final_agent_{j}.wav')
                
                sf.write(final_user_path, user_audio.T, sample_rate)
                sf.write(final_agent_path, agent_audio.T, sample_rate)
                
                # Create recordings and get actual durations
                user_recording = Recording.from_file(final_user_path)
                agent_recording = Recording.from_file(final_agent_path)
                
                # Update cut with actual durations
                cut.recording = user_recording
                cut.target_audio = agent_recording
                cut.duration = user_recording.duration
                cut.duration_no_sil = user_recording.duration - turn_silence_sec
                cut.start = 0.0
                
                # Add the fully processed cut to our list
                all_cuts.append(cut)

            except Exception as e:
                print(f"Error processing cut {j}: {str(e)}")
                continue

        # Create CutSet once after all cuts are fully processed
        print("Creating final CutSet...")
        cuts = CutSet.from_cuts(all_cuts)
        print(f"Created {len(cuts)} cuts")

        # Create shards using to_shar method
        print("Creating shards...")
        cuts.to_shar(
            output_dir,
            fields={"recording": "flac", "target_audio": "flac"},
            num_jobs=num_jobs,
            shard_size=shard_size
        )
        print(f"Created shards in {output_dir}")

    finally:
        # Clean up temporary directory after shards are created
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--answer_audio_dir", type=str, required=True, help="Path to answer audio directory")
    parser.add_argument("--shard_size", type=int, default=100, help="Number of entries per shard")
    parser.add_argument("--shard_prefix", type=str, default="cuts", help="Prefix for shard files")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--turn_silence_sec", type=float, default=0.32, help="Silence duration between turns")
    parser.add_argument("--speaker_after_user", type=str, default="assistant", help="Speaker name after user turn")
    args = parser.parse_args()

    # Create shards using the create_shards function
    create_shards(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        audio_dir=args.audio_dir,
        answer_audio_dir=args.answer_audio_dir,
        shard_size=args.shard_size,
        shard_prefix=args.shard_prefix,
        num_jobs=args.num_jobs,
        random_seed=args.random_seed,
        turn_silence_sec=args.turn_silence_sec,
        speaker_after_user=args.speaker_after_user
    )

if __name__ == "__main__":
    main() 