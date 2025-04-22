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
from io import BytesIO
import torchaudio
import re


torchaudio.set_audio_backend("sox_io")
import lhotse.audio
print(lhotse.audio.audio_backend)


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(manifest, out_shar_dir, num_shard=10, overlap_sec=0.64, audio_dir=None, silent_duration=5.0):
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")
    shard_size = int(len(in_manifest) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")
    print(f"silent_duration: {silent_duration} seconds")

    user_recordings = []
    answer_list = []
    instructions = []
    source_language = []
    target_language = []
    target_recordings = []
    audio_ids = []  # Store original audio IDs
    
    # Process valid entries from the manifest
    valid_manifest = []
    for i, line in tqdm(enumerate(in_manifest)):
        # VoiceBench manifests have one user turn per entry
        if "conversations" in line and len(line["conversations"]) > 0:
            valid_manifest.append(line)

    print(f"total entries: {len(in_manifest)}")
    print(f"valid entries: {len(valid_manifest)}")

    for i, line in tqdm(enumerate(valid_manifest)):
        # Get conversations from the manifest
        convs = line["conversations"]
        
        # Process user turn (first item in conversations)
        user_conv = convs[0]
        
        # Verify it's a user turn
        if user_conv["from"] == "user":
            # Extract the audio path
            audio_path = user_conv["audio_value"]
            
            # Determine full audio path - handle paths starting with "recordings/"
            if audio_path.startswith("recordings/"):
                # Strip the "recordings/" prefix if present, since we're already using AUDIO_DIR as the recordings dir
                audio_file = audio_path.replace("recordings/", "", 1)
                full_audio_path = os.path.join(audio_dir, audio_file)
            elif not os.path.isabs(audio_path):
                # If it's relative (but not starting with "recordings/"), join with audio_dir
                full_audio_path = os.path.join(audio_dir, audio_path)
            else:
                # If it's absolute, use it as is
                full_audio_path = audio_path
            
            # Load the recording if the file exists
            if os.path.exists(full_audio_path):
                # Extract ID from audio filename - get just the numeric part
                audio_filename = os.path.basename(full_audio_path)
                # Extract the ID from the filename (e.g., "1265-34.wav" -> "34")
                match = re.search(r'(\d+)(?:-(\d+))?\.wav$', audio_filename)
                if match and match.group(2):
                    # If format is like "1265-34.wav", use "34"
                    audio_id = match.group(2)
                elif match:
                    # If format is like "1265.wav", use "1265"
                    audio_id = match.group(1)
                else:
                    # Fallback: use the whole filename without extension
                    audio_id = os.path.splitext(audio_filename)[0]
                
                # Create recording with the custom ID
                user_recording = Recording.from_file(full_audio_path)
                # Store the original ID for later use
                audio_ids.append(audio_id)
                user_recordings.append(user_recording)
                
                # Set language (default to English if not specified)
                if "language" in user_conv:
                    source_language.append(user_conv["language"])
                else:
                    source_language.append("EN")
                
                # Add assistant response if present (for conversations with 2+ turns)
                if len(convs) > 1 and convs[1]["from"] == "assistant":
                    assistant_conv = convs[1]
                    # Add text for the assistant response
                    if "text_value" in assistant_conv:
                        answer_list.append(assistant_conv["text_value"])
                    elif "normalized_text_value" in assistant_conv:
                        answer_list.append(assistant_conv["normalized_text_value"])
                    else:
                        answer_list.append("")
                        
                    # Set target language
                    if "language" in assistant_conv:
                        target_language.append(assistant_conv["language"])
                    else:
                        target_language.append("EN")
                else:
                    # If no assistant response, add empty placeholders
                    answer_list.append("")
                    target_language.append("EN")
            else:
                print(f"Warning: Audio file not found: {full_audio_path}")

    print("Done extracting data from manifest")
    print(f"Processed recordings: {len(user_recordings)}")
    
    # Create CutSet from the recordings
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))

    # Attach text and metadata to each cut
    for j, cut in tqdm(enumerate(cuts)):
        if j >= len(valid_manifest):
            break
        
        # Get the originally extracted audio ID
        audio_id = audio_ids[j]
        
        # Set the cut ID to match the extracted audio ID
        cut.id = audio_id
        
        convs = valid_manifest[j]["conversations"]
        user_conv = convs[0]
        
        # Get user text
        user_text = ""
        if "text_value" in user_conv:
            user_text = user_conv["text_value"]
        elif "normalized_text_value" in user_conv:
            user_text = user_conv["normalized_text_value"]
        elif "prompt" in user_conv:
            user_text = user_conv["prompt"]
        
        # Get the recording sample rate and user audio
        sample_rate = cut.recording.sampling_rate
        user_duration = cut.duration
        user_audio = cut.recording.load_audio()
        
        # Create a silent audio array for assistant turn (customizable duration)
        silence_duration_seconds = silent_duration  # Using the provided duration
        silent_audio = np.zeros((1, int(silence_duration_seconds * sample_rate)))  # silent audio with custom duration
        
        # Get assistant text if available
        assistant_text = ""
        if len(convs) > 1 and convs[1]["from"] == "assistant":
            if "text_value" in convs[1]:
                assistant_text = convs[1]["text_value"]
            elif "normalized_text_value" in convs[1]:
                assistant_text = convs[1]["normalized_text_value"]
            elif "reference" in convs[1]:
                assistant_text = convs[1]["reference"]
        
        # Concatenate user audio with zeros for assistant position (following BFCL pattern)
        # For recording: concatenate user audio with zeros for assistant position
        recording_audio = np.concatenate([user_audio, silent_audio], axis=1)
        
        # For target_audio: zeros for user position, silent audio for assistant position
        target_audio = np.concatenate([np.zeros_like(user_audio), silent_audio], axis=1)
        
        # Save the concatenated audio files with the custom ID
        save_audio(f"/tmp/{audio_id}_1.wav", recording_audio, sample_rate)
        save_audio(f"/tmp/{audio_id}_2.wav", target_audio, sample_rate)
        
        # Update the cut's total duration to include both segments
        cut.duration = user_duration + silence_duration_seconds
        
        # Add assertion to verify duration calculation
        assert cut.duration == user_duration + silence_duration_seconds, f"Cut duration {cut.duration} does not match sum of user duration {user_duration} and assistant duration {silence_duration_seconds}"
        
        # Add user supervision segment
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=user_duration,
                text=user_text,
                speaker="user",
                language="EN",
            ),
        )
        
        # Add assistant supervision segment
        cut.supervisions.append(
            SupervisionSegment(
                id=f"{cut.id}_assistant",
                recording_id=cut.id,
                start=user_duration,  # Place after user speech
                duration=silence_duration_seconds,  # Using the silence duration
                text=assistant_text,  # Will be empty if no assistant turn was found
                speaker="assistant",
                language="EN",
            ),
        )
        
        # Create and update the recording with proper ID
        new_recording = Recording.from_file(f"/tmp/{audio_id}_1.wav")
        new_recording.id = cut.id  # Set ID to match the cut
        cut.recording = new_recording
        
        # Create and update the target_audio with proper ID
        target_recording = Recording.from_file(f"/tmp/{audio_id}_2.wav")
        target_recording.id = f"{cut.id}_target"  # Use a consistent target ID format
        cut.target_audio = target_recording

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to shar format
    exported = cuts.to_shar(out_shar_dir, fields={}, num_jobs=1, shard_size=shard_size)
    print(f"...shars created")
    
    # Create separate tar files for target_audio and recording
    for i, path in tqdm(enumerate(exported["cuts"])):
        # Create target_audio tar file
        out_path = path.replace("cuts", "target_audio").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="wav") as writer:
            for cut in CutSet.from_file(path):
                writer.write(
                    cut.id, cut.target_audio.load_audio(), manifest=cut.target_audio, sampling_rate=sample_rate
                )
        
        # Create recording tar file
        out_path = path.replace("cuts", "recording").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="wav") as writer:
            for cut in CutSet.from_file(path):
                writer.write(cut.id, cut.recording.load_audio(), manifest=cut.recording, sampling_rate=sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to the VoiceBench manifest file',
    )
    parser.add_argument(
        '--out_shar_dir',
        type=str,
        required=True,
        help='Output directory for the shards',
    )
    parser.add_argument(
        '--num_shard',
        type=int,
        default=10,
        help='Number of shards to create',
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.64,
        help='Overlap in seconds between turns (if applicable)',
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default=None,
        help='Base directory for audio files (optional)',
    )
    parser.add_argument(
        '--silent_duration',
        type=float,
        default=5.0,
        help='Duration in seconds for the silent assistant segment (default: 5.0)',
    )

    args = parser.parse_args()
    print(f"manifest: {args.manifest}")
    print(f"out_shar_dir: {args.out_shar_dir}")
    print(f"num_shard: {args.num_shard}")
    print(f"overlap: {args.overlap}")
    print(f"audio_dir: {args.audio_dir}")
    print(f"silent_duration: {args.silent_duration}")

    create_shar_from_manifest(
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
        overlap_sec=args.overlap,
        audio_dir=args.audio_dir,
        silent_duration=args.silent_duration
    )


if __name__ == "__main__":
    main() 