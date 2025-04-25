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
    # Import numpy locally to avoid reference errors
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    
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
                # Extract the ID from the filename
                
                # Handle special case for AlpacaEval and other datasets with speaker_id in the name
                if "alpacaeval_speaker" in manifest.lower() or "sd-qa" in manifest.lower():
                    # For files like "en_US_Wavenet_A_1.0_0.0_0.0_60.wav", keep the whole name as ID (without extension)
                    audio_id = os.path.splitext(audio_filename)[0]
                    print(f"Using full ID for speaker dataset: {audio_id}")
                else:
                    # For other datasets, extract just the numeric part
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
        
        # Create temporary WAV files for user and model parts
        user_part = user_audio[: int(user_duration * sample_rate)]
        model_part = np.zeros((user_audio.shape[0], int(silence_duration_seconds * sample_rate)))
        
        # Make safe filenames by removing problematic characters
        safe_id = str(audio_id).replace("/", "_").replace("\\", "_").replace(":", "_")
        
        # Create temporary WAV files with full ID in the filename
        tmp_wav_path1 = f"/tmp/{safe_id}_1.wav"
        tmp_wav_path2 = f"/tmp/{safe_id}_2.wav"
        
        # Save user part and silence as separate files
        sf.write(tmp_wav_path1, np.concatenate([user_part, model_part], axis=1).T, sample_rate)
        sf.write(tmp_wav_path2, np.concatenate([user_part, model_part], axis=1).T, sample_rate)
        
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
        new_recording = Recording.from_file(tmp_wav_path1)
        new_recording.id = cut.id  # Set ID to match the cut
        cut.recording = new_recording
        
        # Create and update the target_audio with proper ID
        target_recording = Recording.from_file(tmp_wav_path2)
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
                try:
                    # Try to write the target audio as normal
                    writer.write(
                        cut.id, cut.target_audio.load_audio(), manifest=cut.target_audio, sampling_rate=sample_rate
                    )
                except Exception as e:
                    print(f"Error loading target audio for {cut.id}: {e}")
                    print(f"Attempting to fix the target audio...")
                    
                    # For SD-QA dataset, we need special handling
                    if "sd-qa" in manifest.lower() or "alpacaeval_speaker" in manifest.lower():
                        try:
                            # Import numpy and soundfile here to ensure they're available
                            import numpy as np
                            import soundfile as sf
                            from pathlib import Path
                            
                            # Create fixed silent audio
                            silence_samples = int(silent_duration * sample_rate)
                            silence = np.zeros((1, silence_samples), dtype=np.float32)
                            
                            # Save to a temporary file with correct properties
                            temp_dir = Path("/tmp")
                            temp_dir.mkdir(exist_ok=True)
                            temp_file = temp_dir / f"{cut.id}_fixed_target.wav"
                            sf.write(str(temp_file), silence.T, sample_rate)
                            
                            # Load from the fixed file
                            fixed_target = Recording.from_file(str(temp_file))
                            fixed_target.id = f"{cut.id}_target"
                            
                            # Write with the fixed audio
                            writer.write(
                                cut.id, fixed_target.load_audio(), manifest=fixed_target, sampling_rate=sample_rate
                            )
                            print(f"Successfully fixed target audio for {cut.id}")
                        except Exception as fix_error:
                            print(f"Failed to fix target audio for {cut.id}: {fix_error}")
                            # Continue to next cut
                            continue
        
        # Create recording tar file
        out_path = path.replace("cuts", "recording").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="wav") as writer:
            for cut in CutSet.from_file(path):
                try:
                    writer.write(cut.id, cut.recording.load_audio(), manifest=cut.recording, sampling_rate=sample_rate)
                except Exception as e:
                    print(f"Error loading recording audio for {cut.id}: {e}")
                    print(f"Attempting to fix the recording audio...")
                    
                    # For SD-QA dataset, we need special handling
                    if "sd-qa" in manifest.lower() or "alpacaeval_speaker" in manifest.lower():
                        try:
                            # Imports should be available from the previous section
                            import numpy as np
                            import soundfile as sf
                            from pathlib import Path
                            
                            # Get the recording duration
                            if cut.recording and hasattr(cut.recording, 'duration') and cut.recording.duration:
                                user_duration = cut.recording.duration
                            else:
                                # Default to a reasonable duration if we can't determine it
                                user_duration = 3.0
                            
                            # Create fixed audio file
                            user_samples = int(user_duration * sample_rate)
                            silence_samples = int(silent_duration * sample_rate)
                            
                            # For recording, we need user audio and silence for assistant
                            recording_samples = user_samples + silence_samples
                            # Create audio with the right shape (silence for user part too)
                            recording_audio = np.zeros((1, recording_samples), dtype=np.float32)
                            
                            # Save to temporary file
                            temp_dir = Path("/tmp")
                            temp_dir.mkdir(exist_ok=True)
                            
                            # Make safe filename with the full ID
                            safe_id = str(cut.id).replace("/", "_").replace("\\", "_").replace(":", "_")
                            temp_file = temp_dir / f"{safe_id}_fixed_recording.wav"
                            
                            sf.write(str(temp_file), recording_audio.T, sample_rate)
                            
                            # Create new recording from the fixed file
                            fixed_recording = Recording.from_file(str(temp_file))
                            fixed_recording.id = cut.id
                            
                            # Write with the fixed audio
                            writer.write(
                                cut.id, fixed_recording.load_audio(), manifest=fixed_recording, sampling_rate=sample_rate
                            )
                            print(f"Successfully fixed recording audio for {cut.id}")
                        except Exception as fix_error:
                            print(f"Failed to fix recording audio for {cut.id}: {fix_error}")
                            # Continue to next cut
                            continue


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