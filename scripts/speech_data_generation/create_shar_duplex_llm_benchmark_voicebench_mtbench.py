import argparse
import json
import os
from pathlib import Path

import numpy as np
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment
from lhotse.audio import save_audio
from lhotse.shar.writers import AudioTarWriter
from tqdm import tqdm


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(manifest, out_shar_dir, num_shard=10, overlap_sec=0.64, audio_dir=None, silent_duration=5.0):
    """
    Create SHAR format data from MTBench manifest with multi-turn conversations.
    
    Args:
        manifest: Path to MTBench manifest file
        out_shar_dir: Output directory for SHAR files
        num_shard: Number of shards to create
        overlap_sec: Overlap in seconds (not used for MTBench)
        audio_dir: Base directory for audio files
        silent_duration: Duration of silent segments for assistant turns
    """
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")
    shard_size = int(len(in_manifest) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")
    print(f"silent_duration: {silent_duration} seconds")

    # Create list to store processed recordings
    all_recordings = []
    valid_manifest = []
    
    # Create a list to store all cuts
    all_cuts = []
    
    print("Processing manifest entries...")
    for i, line in tqdm(enumerate(in_manifest)):
        # MTBench has conversations field with multiple turns
        if "conversations" in line and len(line["conversations"]) > 0:
            valid_manifest.append(line)
    
    print(f"total entries: {len(in_manifest)}")
    print(f"valid entries: {len(valid_manifest)}")

    # Process each conversation
    for i, line in tqdm(enumerate(valid_manifest), desc="Processing conversations"):
        # Get conversations from the manifest
        convs = line["conversations"]
        
        # Skip if no conversations
        if len(convs) == 0:
            continue
        
        # Create a list to store all segments for this conversation
        all_segments = []
        total_duration = 0
        
        # Keep track of current user recording for handling assistant segments
        current_user_recording = None
        
        # Get conversation ID
        conv_id = line.get("id", str(i))
        
        # Debug info
        print(f"Processing conversation ID: {conv_id}")
        
        # Process turns in order they appear (user, assistant, user, assistant...)
        for turn_idx, turn in enumerate(convs):
            # Skip turns without proper role
            if "from" not in turn:
                continue
                
            # Process user turn with audio
            if turn["from"] == "user":
                # Get audio path 
                if "audio_value" not in turn:
                    print(f"Warning: User turn {turn_idx} has no audio_value, skipping")
                    continue
                    
                audio_path = turn["audio_value"]
                
                # Use the exact path from the manifest
                if audio_path.startswith("recordings/"):
                    # This is a relative path - join with audio_dir
                    audio_file = audio_path.replace("recordings/", "", 1)
                    full_audio_path = os.path.join(audio_dir, audio_file)
                elif not os.path.isabs(audio_path):
                    # Any other relative path
                    full_audio_path = os.path.join(audio_dir, audio_path)
                else:
                    # Absolute path
                    full_audio_path = audio_path
                
                # Check if file exists
                if os.path.exists(full_audio_path):
                    try:
                        user_recording = Recording.from_file(full_audio_path)
                        current_user_recording = user_recording  # Save for assistant turns
                        print(f"Successfully loaded audio: {full_audio_path}")
                        
                        # Get user text
                        user_text = ""
                        if "text_value" in turn:
                            user_text = turn["text_value"]
                        elif "normalized_text_value" in turn:
                            user_text = turn["normalized_text_value"]
                        elif "prompt" in turn:
                            user_text = turn["prompt"]
                        
                        # Add user segment
                        user_duration = user_recording.duration
                        
                        # Add user segment to the list in order
                        all_segments.append({
                            "audio": user_recording.load_audio(),
                            "duration": user_duration,
                            "text": user_text,
                            "speaker": "user",
                            "turn_idx": turn_idx
                        })
                        
                        total_duration += user_duration
                    except Exception as e:
                        print(f"Error loading audio {full_audio_path}: {str(e)}")
                        continue
                else:
                    print(f"Warning: Audio file not found: {full_audio_path}")
                    continue
            
            # Process assistant turn - must come after a user turn
            elif turn["from"] == "assistant":
                if current_user_recording is None:
                    print(f"Warning: Assistant turn {turn_idx} has no preceding user turn, skipping")
                    continue
                    
                # Get assistant text
                assistant_text = ""
                if "text_value" in turn:
                    assistant_text = turn["text_value"]
                elif "normalized_text_value" in turn:
                    assistant_text = turn["normalized_text_value"]
                elif "reference" in turn:
                    assistant_text = turn["reference"]
                
                # Use silent duration for all assistant turns
                sample_rate = current_user_recording.sampling_rate
                silent_audio = np.zeros((1, int(silent_duration * sample_rate)))
                
                # Add assistant segment to the list with silent audio
                all_segments.append({
                    "audio": silent_audio,
                    "duration": silent_duration,
                    "text": assistant_text,  # Can be empty or contain text
                    "speaker": "assistant",
                    "turn_idx": turn_idx
                })
                
                total_duration += silent_duration
        
        # Skip if we didn't get any valid segments
        if len(all_segments) == 0:
            continue
            
        # Sort segments by turn_idx to ensure correct order
        all_segments.sort(key=lambda x: x["turn_idx"])
            
        # Create concatenated audio for the entire conversation
        sample_rate = current_user_recording.sampling_rate
        recording_audio = np.zeros((1, 0))
        target_audio = np.zeros((1, 0))
        
        # Process each segment and concatenate audio
        supervisions = []
        current_start = 0
        expected_duration = total_duration  # Initialize expected_duration with original total duration
        
        for seg_idx, segment in enumerate(all_segments):
            # Concatenate audio based on speaker
            if segment["speaker"] == "user":
                # For recording: use user audio
                recording_audio = np.concatenate([recording_audio, segment["audio"]], axis=1)
                # For target: use silence of same length
                target_audio = np.concatenate([target_audio, np.zeros_like(segment["audio"])], axis=1)
            else:
                # For recording: use silence for assistant position
                recording_audio = np.concatenate([recording_audio, segment["audio"]], axis=1)
                # For target: use silence for assistant position (same as recording)
                target_audio = np.concatenate([target_audio, segment["audio"]], axis=1)
            
            # Create supervision segment with unique ID for the turn
            sup_id = f"{conv_id}_{seg_idx}"
            
            # Add supervision segment
            supervisions.append(
                SupervisionSegment(
                    id=sup_id,
                    recording_id=conv_id,
                    start=current_start,
                    duration=segment["duration"],
                    text=segment["text"],
                    speaker=segment["speaker"],
                    language="EN",
                )
            )
            
            # Update start time for next segment
            current_start += segment["duration"]
            
        # Check if the last segment is from a user - if so, add a silent assistant response
        if all_segments and all_segments[-1]["speaker"] == "user":
            # Add a silent assistant segment after the last user segment
            silence_duration = silent_duration  # Use the provided silent duration
            silent_audio = np.zeros((1, int(silence_duration * sample_rate)))
            
            # Concatenate silent audio
            recording_audio = np.concatenate([recording_audio, silent_audio], axis=1)
            target_audio = np.concatenate([target_audio, silent_audio], axis=1)
            
            # Create supervision segment for the silent assistant response
            sup_id = f"{conv_id}_{len(all_segments)}"
            supervisions.append(
                SupervisionSegment(
                    id=sup_id,
                    recording_id=conv_id,
                    start=current_start,
                    duration=silence_duration,
                    # No text for silent segment
                    speaker="assistant",
                    language="EN",
                )
            )
            
            # Update current_start for completeness
            current_start += silence_duration
            
            # Important: Update the expected duration with the added silence
            expected_duration += silence_duration
        
        # Save concatenated audio
        save_audio(f"/tmp/{conv_id}_1.wav", recording_audio, sample_rate)
        save_audio(f"/tmp/{conv_id}_2.wav", target_audio, sample_rate)
        
        # Create cut for this conversation
        recording = Recording.from_file(f"/tmp/{conv_id}_1.wav", recording_id=conv_id)
        target_audio_rec = Recording.from_file(f"/tmp/{conv_id}_2.wav", recording_id=f"{conv_id}_target")
        
        # Add the recording to our list
        all_recordings.append(recording)
        
        # Create cut with all supervision segments
        cut = recording.to_cut()
        cut.supervisions = supervisions
        cut.target_audio = target_audio_rec
        
        # Verify the cut duration matches the sum of all segment durations
        assert abs(cut.duration - expected_duration) < 0.001, f"Cut duration {cut.duration} does not match expected duration {expected_duration}"
        
        # Add the cut to our list of cuts
        all_cuts.append(cut)

    # After the loop, create a CutSet from all cuts
    if all_cuts:
        cuts = CutSet.from_cuts(all_cuts)
    else:
        print("No valid cuts were created. All audio files might be missing. Exiting.")
        return

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to shar format
    exported = cuts.to_shar(out_shar_dir, fields={}, num_jobs=1, shard_size=shard_size)
    print(f"...shars created")
    
    # Create separate tar files for target_audio and recording
    sample_rate = all_recordings[0].sampling_rate if all_recordings else 16000
    for i, path in tqdm(enumerate(exported["cuts"]), desc="Creating audio tars"):
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
        default='/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM_LLM_Benchmarks/VoiceBench/mtbench/manifests/mtbench_manifest.json',
        help='Path to the MTBench manifest file',
    )
    parser.add_argument(
        '--out_shar_dir',
        type=str,
        default='/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM_LLM_Benchmarks/VoiceBench/mtbench/shars',
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
        help='Overlap in seconds between turns (not used for MTBench)',
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM_LLM_Benchmarks/VoiceBench/mtbench/recordings',
        help='Base directory for audio files',
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