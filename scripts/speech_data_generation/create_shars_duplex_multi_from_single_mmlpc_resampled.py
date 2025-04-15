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

def get_audio(recording, sample_rate):
    # Resample the audio to the target sample rate
    audio_data = recording.load_audio()
    if recording.sampling_rate != sample_rate:
        audio_data = librosa.resample(audio_data, orig_sr=recording.sampling_rate, target_sr=sample_rate)
    return audio_data

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
        # Read manifest and process entries
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
                continue
                
        in_manifest = cleaned_manifest
        print(f"Number of valid recordings: {len(in_manifest)}")
        print(f"Number of excluded German transcription examples: {excluded_german}")
        print(f"Number of excluded examples due to missing files: {excluded_missing_files}")
        print(f"Total excluded examples: {excluded_german + excluded_missing_files}")

        # Create CutSet with proper fields
        print("Creating CutSet...")
        cuts = []
        
        # Process entries taking 2 from start and 2 from end
        num_entries = len(in_manifest)
        num_cuts = num_entries // 4  # Each cut needs 4 entries
        
        print(f"Will create {num_cuts} cuts, processing 4 samples at a time")
        cuts = []
        
        # Add progress bar for cut creation
        for i in tqdm(range(num_cuts), desc="Creating cuts"):
            try:
                # Get 2 entries from start and 2 from end
                start_idx = i * 2
                end_idx = num_entries - (i * 2) - 2
                
                if end_idx <= start_idx:
                    break  # Stop when we've processed all entries
                    
                group = in_manifest[start_idx:start_idx+2] + in_manifest[end_idx:end_idx+2]
                if len(group) != 4:
                    continue
                    
                # Create recordings for all files in the group
                recordings = []
                target_recordings = []
                for idx, entry in enumerate(group):
                    # Get paths
                    audio_wav = os.path.join(audio_dir, entry["audio_filepath"])
                    question_wav = os.path.join(audio_dir, entry["question_wav"])
                    target_wav = os.path.join(answer_audio_dir, entry["target_wav"])
                    
                    # Load and concatenate user audio (audio_wav + question_wav)
                    user_audio = Recording.from_file(audio_wav)
                    question_audio = Recording.from_file(question_wav)
                    
                    # Load audio data
                    user_audio_data = user_audio.load_audio()
                    question_audio_data = question_audio.load_audio()
                    
                    # Resample user audio to match question audio's sampling rate
                    if user_audio.sampling_rate != question_audio.sampling_rate:
                        user_audio_data = librosa.resample(
                            user_audio_data.squeeze(),
                            orig_sr=user_audio.sampling_rate,
                            target_sr=question_audio.sampling_rate
                        ).reshape(1, -1)
                    
                    # Concatenate without silence
                    combined_audio = np.concatenate([user_audio_data, question_audio_data], axis=1)
                    
                    # Save concatenated audio
                    temp_user_path = os.path.join(temp_dir, f"user_{i}_{idx}.wav")
                    sf.write(temp_user_path, combined_audio.T, question_audio.sampling_rate)
                    
                    # Create recording from concatenated audio
                    recordings.append(Recording.from_file(temp_user_path))
                    target_recordings.append(Recording.from_file(target_wav))
                
                # Concatenate all user recordings with silence
                recording_audio = []
                target_audio = []
                for rec, target_rec in zip(recordings, target_recordings):
                    # Load audio data
                    rec_audio = rec.load_audio()
                    target_audio_data = target_rec.load_audio()
                    
                    # Add silence
                    silence = np.zeros((1, int(turn_silence_sec * rec.sampling_rate)))
                    
                    # Concatenate with zeros in inactive channel
                    recording_audio.append(rec_audio)
                    recording_audio.append(silence)
                    recording_audio.append(np.zeros_like(target_audio_data))
                    
                    target_audio.append(np.zeros_like(rec_audio))
                    target_audio.append(silence)
                    target_audio.append(target_audio_data)
                
                # Combine all audio segments
                final_recording_audio = np.concatenate(recording_audio, axis=1)
                final_target_audio = np.concatenate(target_audio, axis=1)
                
                # Create temporary files for concatenated audio
                temp_recording_path = os.path.join(temp_dir, f"recording_{i}.wav")
                temp_target_path = os.path.join(temp_dir, f"target_{i}.wav")
                
                # Save concatenated audio
                sf.write(temp_recording_path, final_recording_audio.T, recordings[0].sampling_rate)
                sf.write(temp_target_path, final_target_audio.T, recordings[0].sampling_rate)
                
                # Create recordings
                final_recording = Recording.from_file(temp_recording_path)
                final_target = Recording.from_file(temp_target_path)
                
                # Create the cut
                cut = MonoCut(
                    id=f"cut_{i}",
                    start=0,
                    duration=final_recording.duration,
                    channel=0,
                    recording=final_recording,
                    supervisions=[]
                )
                
                # Add supervisions for each turn with proper timing
                current_time = 0
                for j, (entry, rec, target_rec) in enumerate(zip(group, recordings, target_recordings)):
                    # Get question and answer text
                    question_text = entry.get("question", "")
                    answer_text = entry.get("answer", "")
                    
                    # Add user turn (audio + question)
                    cut.supervisions.append(
                        SupervisionSegment(
                            id=entry["audio_filepath"],
                            recording_id=entry["audio_filepath"],
                            start=current_time,
                            duration=rec.duration,
                            channel=0,
                            text=f"{entry['text']} {question_text}".strip(),  # Concatenate audio text and question text
                            language="en",
                            speaker="user"
                        )
                    )
                    current_time += rec.duration
                    
                    # Add silence
                    current_time += turn_silence_sec
                    
                    # Add assistant turn
                    cut.supervisions.append(
                        SupervisionSegment(
                            id=entry["target_wav"],
                            recording_id=entry["target_wav"],
                            start=current_time,
                            duration=target_rec.duration,
                            channel=0,
                            text=answer_text,
                            language="en",
                            speaker="assistant"
                        )
                    )
                    current_time += target_rec.duration
                    
                    # Add silence between pairs
                    if j < len(group) - 1:
                        current_time += turn_silence_sec
                
                # Calculate the last assistant's end time
                last_assistant_end = 0.0
                for supervision in cut.supervisions:
                    if supervision.speaker == speaker_after_user:  # This is an assistant turn
                        end_time = supervision.start + supervision.duration
                        if end_time > last_assistant_end:
                            last_assistant_end = end_time
                
                # Calculate required duration in samples
                required_duration_samples = int(last_assistant_end * recordings[0].sampling_rate)
                
                # Pad both audio files to match the required duration
                if final_recording_audio.shape[1] < required_duration_samples:
                    padding = np.zeros((1, required_duration_samples - final_recording_audio.shape[1]))
                    final_recording_audio = np.concatenate([final_recording_audio, padding], axis=1)
                if final_target_audio.shape[1] < required_duration_samples:
                    padding = np.zeros((1, required_duration_samples - final_target_audio.shape[1]))
                    final_target_audio = np.concatenate([final_target_audio, padding], axis=1)
                
                # Save padded audio
                sf.write(temp_recording_path, final_recording_audio.T, recordings[0].sampling_rate)
                sf.write(temp_target_path, final_target_audio.T, recordings[0].sampling_rate)
                
                # Update recordings with padded audio
                final_recording = Recording.from_file(temp_recording_path)
                final_target = Recording.from_file(temp_target_path)
                
                # Update cut with new duration and recording
                cut.duration = last_assistant_end
                cut.recording = final_recording
                
                # Set target_audio in custom field
                cut.custom = {
                    "target_audio": final_target,
                    "duration_no_sil": last_assistant_end - (len(group) * turn_silence_sec)
                }
                
                # Add type field
                cut.type = "MonoCut"
                
                cuts.append(cut)
                
            except Exception as e:
                print(f"Error processing cut {i}: {str(e)}")
                continue
        
        # Create CutSet
        cutset = CutSet.from_cuts(cuts)
        print(f"Created CutSet with {len(cutset)} cuts")

        # Save the cutset to shards
        print("...Making Shars")
        out_shar_dir = Path(output_dir)
        out_shar_dir.mkdir(parents=True, exist_ok=True)
        
        # Create shards with the correct file structure
        print(f"Creating {len(cutset) // shard_size + 1} shards...")
        exported = cutset.to_shar(
            out_shar_dir,
            fields={"recording": "flac", "target_audio": "flac"},
            num_jobs=1,
            shard_size=shard_size
        )
        print(f"...share created")

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--answer_audio_dir", type=str, required=True, help="Path to answer audio directory")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of cuts per shard")
    parser.add_argument("--shard_prefix", type=str, default="cuts", help="Prefix for shard files")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of jobs for parallel processing")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--turn_silence_sec", type=float, default=0.32, help="Silence duration between turns")
    parser.add_argument("--speaker_after_user", type=str, default="assistant", help="Speaker after user turn")
    args = parser.parse_args()

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
        speaker_after_user=args.speaker_after_user,
    )

if __name__ == "__main__":
    main() 