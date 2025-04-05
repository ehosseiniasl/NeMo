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
from lhotse.shar import SharWriter
from lhotse.shar.writers.array import ArrayTarWriter
from lhotse.shar.writers.audio import AudioTarWriter
from lhotse.shar.writers.text import TextTarWriter
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
) -> None:
    """Create shards from a manifest with multiple speakers."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest
    in_manifest = list(json_reader(manifest_path))
    print(f"Read {len(in_manifest)} entries from manifest")

    # Check if required files exist and clean manifest
    cleaned_manifest = []
    for i, entry in tqdm(enumerate(in_manifest)):
        try:
            # Get audio paths and construct full paths
            audio_filepath = entry["audio_filepath"]
            question_wav = os.path.join(audio_dir, audio_filepath)
            target_wav = os.path.join(answer_audio_dir, entry["target_wav"])
            
            # Check if required files exist
            if not os.path.exists(question_wav):
                raise FileNotFoundError(f"File not found: {question_wav}")
            if not os.path.exists(target_wav):
                raise FileNotFoundError(f"File not found: {target_wav}")
                
            cleaned_manifest.append(entry)
            
        except Exception as e:
            logging.info(f'Skipping {i}th json record: {str(e)}')
            
    in_manifest = cleaned_manifest
    print(f"Number of valid recordings: {len(in_manifest)}")

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
            temp_files = []  # Keep track of temporary files to clean up later
            for entry_idx, entry in enumerate(entries):
                try:
                    # Get audio paths and construct full paths
                    audio_filepath = entry["audio_filepath"]
                    question_wav = os.path.join(audio_dir, audio_filepath)
                    target_wav = os.path.join(answer_audio_dir, entry["target_wav"])
                    
                    # Load and concatenate both audio files
                    user_audio = Recording.from_file(question_wav)
                    question_audio = Recording.from_file(question_wav)
                    
                    # Concatenate the recordings (user_audio first, then question_audio)
                    user_recording = Recording.concatenate([user_audio, question_audio])
                    agent_recording = Recording.from_file(target_wav)
                    
                    # Create supervision segments
                    user_supervision = SupervisionSegment(
                        id=os.path.basename(question_wav),
                        recording_id=os.path.basename(question_wav),
                        start=total_dur,
                        duration=user_recording.duration,
                        text=f"{entry.get('text', '')} {entry.get('question', '')}",
                        speaker="USER",
                        language=entry.get("source_lang", "EN"),
                    )
                    
                    agent_supervision = SupervisionSegment(
                        id=os.path.basename(target_wav),
                        recording_id=os.path.basename(target_wav),
                        start=total_dur + user_recording.duration + turn_silence_sec,
                        duration=agent_recording.duration,
                        text=entry.get("answer", ""),
                        speaker=entry.get("answer_speaker", "AGENT"),
                        language=entry.get("target_lang", "EN"),
                    )

                    # Add supervisions to cut
                    cut.supervisions.append(user_supervision)
                    cut.supervisions.append(agent_supervision)

                    # Process audio
                    sample_rate = agent_recording.sampling_rate
                    user_duration = user_recording.duration + turn_silence_sec
                    agent_duration = agent_recording.duration
                    cur_user_audio = user_recording.resample(sample_rate).load_audio()
                    cur_agent_audio = agent_recording.load_audio()

                    silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
                    user_audio_list.extend([cur_user_audio, silence_padding, np.zeros_like(cur_agent_audio)])
                    agent_audio_list.extend([np.zeros_like(cur_user_audio), silence_padding, cur_agent_audio])

                    total_dur += user_duration + agent_duration

                except Exception as e:
                    print(f"Error processing entry {entry_idx} in cut {j}: {str(e)}")
                    continue

            # Clean up temporary files after we're done with this cut
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logging.info(f"Failed to remove temporary file {temp_file}: {str(e)}")

            # Process final audio
            user_audio = np.concatenate(user_audio_list, axis=1)
            agent_audio = np.concatenate(agent_audio_list, axis=1)
            
            cut.duration = total_dur + turn_silence_sec
            cut.duration_no_sil = total_dur
            cut.start = 0.0

            # Save user and agent audio to temporary files
            user_temp_path = f"/tmp/final_user_{j}.wav"
            agent_temp_path = f"/tmp/final_agent_{j}.wav"
            
            # Save the audio using soundfile
            sf.write(user_temp_path, user_audio.T, sample_rate)
            sf.write(agent_temp_path, agent_audio.T, sample_rate)
            
            # Create recordings from the temporary files
            cut.recording = Recording.from_file(user_temp_path)
            cut.target_audio = Recording.from_file(agent_temp_path)
            
            # Clean up temporary files
            try:
                os.remove(user_temp_path)
                os.remove(agent_temp_path)
            except Exception as e:
                logging.info(f"Failed to remove temporary files: {str(e)}")
                
            # Add the fully processed cut to our list
            all_cuts.append(cut)

        except Exception as e:
            print(f"Error processing cut {j}: {str(e)}")
            continue

    # Create CutSet once after all cuts are fully processed
    print("Creating final CutSet...")
    cuts = CutSet.from_cuts(all_cuts)
    print(f"Created {len(cuts)} cuts")

    # Create shards
    with SharWriter(
        output_dir,
        shard_size=shard_size,
        shard_prefix=shard_prefix,
        fields={
            "recording": AudioTarWriter,
            "target_audio": AudioTarWriter,
            "features": ArrayTarWriter,
            "custom_fields": TextTarWriter,
        },
    ) as writer:
        for cut in cuts:
            writer.write(cut)

    print(f"Created shards in {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help="Path to the input manifest.",
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help="Directory containing the question audio files.",
    )
    parser.add_argument(
        '--answer_audio_dir',
        type=str,
        required=True,
        help="Directory containing the answer audio files.",
    )
    parser.add_argument(
        '--shard_size',
        type=int,
        default=1000,
        help="Number of cuts per shard.",
    )
    parser.add_argument(
        '--shard_prefix',
        type=str,
        default="cuts",
        help="Prefix for the shard files.",
    )
    parser.add_argument(
        '--num_jobs',
        type=int,
        default=1,
        help="Number of jobs for parallel processing.",
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        '--turn_silence_sec',
        type=float,
        default=0.32,
        help="Silence duration between turns in seconds.",
    )

    args = parser.parse_args()
    create_shards(
        args.manifest,
        args.output_dir,
        args.audio_dir,
        args.answer_audio_dir,
        args.shard_size,
        args.shard_prefix,
        args.num_jobs,
        args.random_seed,
        args.turn_silence_sec,
    )

if __name__ == "__main__":
    main() 