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
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm
import ipdb
from io import BytesIO
import tempfile

# from lhotse.audio import set_audio_backend
# set_audio_backend("torchaudio")
# import torchaudio
# torchaudio.set_audio_backend("sox_io")
# import torchaudio
# torchaudio.set_audio_backend("soundfile")
# torchaudio.set_audio_backend("ffmpeg")
# import torchaudio.backend
# torchaudio.set_audio_backend("soundfile")
import os
os.environ["LHOTSE_STRICT"] = "0"

def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def num_function_turns(data):
    c = 0
    for i, d in enumerate(data):
        # if "<TOOLCALL>" in d['value_normalized'] or "<TOOL_RESPONSE>" in d['value_normalized']:
        if i % 2 == 0 and ("<TOOLCALL>" in d['value'] or "<TOOL_RESPONSE>" in d['value']):
            c += 1
        elif i % 2 != 0 and ("<TOOLCALL>" in d['value'] or "<TOOL_RESPONSE>" in d['value']):
            c += 1
    return c

def create_shar_from_manifest(manifest, out_shar_dir, audio_dir, num_shard=10, overlap_sec=0.64):
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
    valid_manifest = []
    for i, line in tqdm(enumerate(in_manifest)):
        # if "audio_value" in line['conversations'][0] and "<AVAILABLE_TOOLS>" not in line['conversations'][0]['value']: # no tool instruction in 1st user turn
        #     fc_calls = num_function_turns(line['conversations'])
        #     print (i, fc_calls)
        #     if fc_calls % 2 == 0:
        #         valid_manifest.append(line)
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

    # audio_dir = "/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/digital_human_alm/data/SFT/ameya_data/synthesized/audio"
    for i, line in tqdm(enumerate(valid_manifest)):
        # For single turn convs is a list of 2 elements
        # First element is user speech and second is agent speech
        convs = line["conversations"]
        # for conv in convs:
        #     if 'value' in conv:
        #         conv["value"] = conv["value"].replace("fs7", "fsw")
        #     if 'normalized_text_value' in conv:
        #         conv["normalized_text_value"] = conv["normalized_text_value"].replace("fs7", "fsw")

        # User_Speech
        # user_recording = Recording.from_file(convs[0]['value'])
        # user_recordings.append(user_recording)
        # found_image = False
        # for turn in convs:
        #     if 'audio_value' in turn:
        #         # if not turn['audio_value'].startswith("/lustre/fsw"):
        #         #     conv_recording = Recording.from_file(os.path.join(audio_dir, turn['audio_value'].split("/")[-1]))
        #         # else:
        #         #     conv_recording = Recording.from_file(turn['audio_value'])
        #         conv_recording = Recording.from_file(os.path.join(audio_dir, turn['audio_value'].split("/")[-1]))
        #         conv_recordings.append(conv_recording)
        #         sample_rate = conv_recording.sampling_rate
        #         found_image = True
        #         break
        # assert found_image == True
        conv_recording = Recording.from_file(os.path.join(audio_dir, convs[0]['audio_value'].split("/")[-1]))
        sample_rate = conv_recording.sampling_rate
        conv_recordings.append(conv_recording)

        # Instructions from the user. In case the question is part of the source audio this is a static text "Transcribe and answer",
        # If not then this is the actual question from the user but in text.
        # For direct_s2s instructions are always empty (else part)
        # if "instruction" in convs[0]:
        #     instructions.append(convs[0]["instruction"])
        # else:
        #     instructions.append("")
        # ipdb.set_trace()
        function = line["conversations"][0]["original_manifest"]["raw_data"]["function"]
        system_prompt = f"<AVAILABLE_TOOLS>{function}</AVAILABLE_TOOLS>"
        instructions.append(system_prompt)

        # Language source
        # if "lang" in convs[0]:
        #     source_language.append(convs[0]["lang"])
        # else:
        source_language.append("EN")

        # Loading agent audio and using only the extracted features as nd.array
        # target_recordings.append(Recording.from_file(convs[1]['value']))
        # Agent answer transcript
        # answer_list.append(convs[1]["transcript"])
        # Language target
        # if "lang" in convs[1]:
        #     target_language.append(convs[1]["lang"])
        # else:
        target_language.append("EN")

    print("Done extracting data from manifest")
    # print(len(user_recordings))
    print(len(conv_recordings))
    # cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(conv_recordings))

    unequal = 0
    # Attach text
    for j, cut in tqdm(enumerate(cuts)):
        turn_silence_sec = 0.32
        silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
        # overlap_sec = overlap #1.28 #0.64
        overlap_samples = int(overlap_sec*sample_rate)

        user_audio = np.array([[]])
        agent_audio = np.array([[]])
        total_dur = 0

        # convs = in_manifest[j]["conversations"]
        convs = valid_manifest[j]["conversations"]
        # cut.target_audios = []
        # cut.source_audios = []
        total_duration = 0
        cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=0,
                    duration=0, #cut.recording.duration,
                    text=instructions[j],
                    speaker="system", 
                    language="EN",
                ),
            )

        cur_user_agent_added = False
        cur_user_audio = None
        cur_agent_audio = None
        # for i in range(len(convs)):

        # if 'audio_value' not in convs[i]:
        # if '<TOOLCALL>' in convs[i]['value'] or '<TOOL_RESPONSE>' in convs[i]['value']:
        #     user_function = convs[i]['value']
        #     user_transcript = ""
        #     user_path = "" #np.zeros((0,0))
        #     user_duration = 0
        # else:
        user_function = ""
        user_transcript = convs[0]['normalized_text_value']
        # if convs[i]['audio_value'].startswith("/lustre/fsw"):
        #     user_path = convs[i]['audio_value']
        # else:
        user_path = os.path.join(audio_dir, convs[0]['audio_value'].split("/")[-1])
        user_duration = Recording.from_file(user_path).duration
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
                speaker="user", #convs[0]["from"],
                language="EN",
            ),
        )
        
        # Assert that supervisions[1] is from 'user'
        assert cut.supervisions[1].speaker == "user", f"Expected supervision[1].speaker to be 'user', got '{cut.supervisions[1].speaker}'"

        # total_duration += user_duration 
                    # ipdb.set_trace()
        # if user_path != '':
        sample_rate = Recording.from_file(user_path).sampling_rate
        try:
            cur_user_audio = Recording.from_file(user_path).resample(sample_rate).load_audio() # ignore metadata in audio
        except:
            ipdb.set_trace()
        cur_user_agent_added = False
        # elif cur_user_agent_added:
        #     cur_user_audio = None
        
        if user_duration > 0:
            total_duration += (user_duration + turn_silence_sec)

        
        # try:
        #     # if 'audio_value' not in convs[i+1]:
        #     if '<TOOLCALL>' in convs[i+1]['value'] or '<TOOL_RESPONSE>' in convs[i+1]['value']:
        #         assistant_function = convs[i+1]['value']
        #         assistant_transcript = ""
        #         assistant_path = "" #np.zeros((0,0))
        #         assistant_duration = 0
        #     else:
        #         assert 'audio_value' in convs[i+1]
        #         assistant_function = ""
        #         assistant_transcript = convs[i+1]['value_normalized']
        #         # assistant_transcript = convs[i+1]['value_summarized']
        #         if convs[i+1]['audio_value'].startswith("/lustre/fsw"):
        #             assistant_path = convs[i+1]['audio_value']
        #         else:
        #             assistant_path = os.path.join(audio_dir, convs[i+1]['audio_value'])
        #         assistant_duration = Recording.from_file(assistant_path).duration
        # except:
        #     import ipdb; ipdb.set_trace()
        
        if "ground_truth" in convs[0]["original_manifest"]["raw_data"]:
            ground_truth = convs[0]["original_manifest"]["raw_data"]["ground_truth"]
            if len(ground_truth) == 1:
                assistant_function = f"<TOOLCALL>{ground_truth}</TOOLCALL>"
            else:
                assistant_function = f"<TOOLCALL>[{','.join(ground_truth)}]</TOOLCALL>"
        else:
            assistant_function = ""
        assistant_transcript = ""
        assistant_path = "" 
        # Assistant duration should be 0 according to the required format
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

        total_duration += assistant_duration

        # Create silent agent audio of the same length as user audio
        cur_agent_audio = np.zeros_like(cur_user_audio)

        if cur_user_audio is not None and cur_agent_audio is not None and not cur_user_agent_added:
            # print(cut.id)
            if i != 0: # overlap this turn with previos one
                if overlap_samples > 0:
                    user_audio = user_audio[:, :-overlap_samples]
                total_dur -= overlap_sec
            
            # Just use the user audio directly without adding additional silence or zeros
            # This ensures the audio duration matches the user's actual speech
            user_audio = np.concatenate([user_audio, cur_user_audio], axis=1)
            
            # For agent audio, just use zeros of the exact same length as user_audio
            if i != 0:
                if overlap_samples > 0:
                    agent_audio = np.concatenate([agent_audio, np.zeros_like(cur_user_audio[:, :-overlap_samples])], axis=1)
                else:
                    agent_audio = np.concatenate([agent_audio, np.zeros_like(cur_user_audio)], axis=1)
            else:
                agent_audio = np.concatenate([agent_audio, np.zeros_like(cur_user_audio)], axis=1)

            cur_user_agent_added = True
            cur_user_audio = None
            cur_agent_audio = None

        # if user_audio != "":
        #     cut.source_audios.append(Recording.from_file(user_audio))
        # else:
        #     cut.source_audios.append(Recording(sources=[], id='', sampling_rate=0, num_samples=1, duration=0))
        
        # if assistant_audio != "":
        #     cut.target_audios.append(Recording.from_file(assistant_audio))
        # else:
        #     cut.target_audios.append(Recording(sources=[], id='', sampling_rate=0, num_samples=1, duration=0))
        
        # Set the cut duration to match the total audio duration
        cut.duration = total_duration
        cut.start = 0.0
        
        # Ensure user and agent audio are of equal length
        try:
            assert user_audio.shape == agent_audio.shape, f"User audio shape {user_audio.shape} != agent audio shape {agent_audio.shape}"
        except:
            unequal += 1
            diff = agent_audio.shape[1] - user_audio.shape[1]
            if diff > 0:
                user_audio = np.concatenate([user_audio, np.zeros((1, int(diff)))], axis=1) # add silence
            else:
                agent_audio = np.concatenate([agent_audio, np.zeros((1, int(np.abs(diff))))], axis=1) # add silence
            # ipdb.set_trace()
            assert user_audio.shape == agent_audio.shape, "Audio shapes still don't match after adjustment"

        # Save the audio files
        save_audio(f"/tmp/u{j}1.wav", user_audio, sample_rate)
        save_audio(f"/tmp/u{j}2.wav", agent_audio, sample_rate)
        
        # Load the recordings and ensure durations match
        user_recording = Recording.from_file(f"/tmp/u{j}1.wav")
        agent_recording = Recording.from_file(f"/tmp/u{j}2.wav")
        
        # Verify that durations match
        assert abs(user_recording.duration - agent_recording.duration) < 1e-6, f"Recording durations don't match: {user_recording.duration} vs {agent_recording.duration}"
        
        cut.recording = user_recording
        cut.target_audio = agent_recording
        
        # IMPORTANT: Set the cut duration to match exactly the recording duration
        # This ensures the overall cut duration is correct
        cut.duration = cut.recording.duration
        
        # Also update total_duration to match the actual recording duration
        total_duration = cut.recording.duration
            
        # Since there are 3 supervisions (system, user, assistant), assistant is at index 2
        # The assistant supervision needs start time updated but duration remains 0
        
        # User supervision should be the exact duration of the user audio
        cut.supervisions[1].duration = cut.recording.duration
        cut.supervisions[1].start = 0  # Ensure user starts at the beginning
        
        # Set assistant start time to the end of the recording
        user_end_time = cut.recording.duration
        
        # Remove the old assistant supervision and add an updated one with duration 0
        old_assistant_supervision = cut.supervisions.pop()
        cut.supervisions.append(
            SupervisionSegment(
                id=old_assistant_supervision.id,
                recording_id=old_assistant_supervision.recording_id,
                start=user_end_time,
                duration=0, # Keep assistant duration as 0
                text=old_assistant_supervision.text,
                custom=old_assistant_supervision.custom,
                speaker=old_assistant_supervision.speaker,
                language=old_assistant_supervision.language,
            ),
        )
        
        # Double check that all the durations are consistent
        assert abs(cut.duration - cut.recording.duration) < 1e-6, f"Cut duration {cut.duration} doesn't match recording duration {cut.recording.duration}"
        assert abs(cut.duration - cut.target_audio.duration) < 1e-6, f"Cut duration {cut.duration} doesn't match target audio duration {cut.target_audio.duration}"
        assert abs(cut.supervisions[1].duration - cut.recording.duration) < 1e-6, f"User supervision duration {cut.supervisions[1].duration} doesn't match recording duration {cut.recording.duration}"
        assert cut.supervisions[1].start == 0, f"User supervision doesn't start at 0: {cut.supervisions[1].start}"
        assert cut.supervisions[2].start == cut.recording.duration, f"Assistant supervision doesn't start at the end of recording: {cut.supervisions[2].start} vs {cut.recording.duration}"
        assert cut.supervisions[2].duration == 0, f"Assistant supervision duration is not 0: {cut.supervisions[2].duration}"
        
        # Print duration details for debugging
        print(f"Cut ID: {cut.id}")
        print(f"Cut duration: {cut.duration}")
        print(f"Recording duration: {cut.recording.duration}")
        print(f"Target audio duration: {cut.target_audio.duration}")
        print(f"User supervision: start={cut.supervisions[1].start}, duration={cut.supervisions[1].duration}")
        print(f"Assistant supervision: start={cut.supervisions[2].start}, duration={cut.supervisions[2].duration}")

    print(f"unequal examples: {unequal}")


    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    shard_size = shard_size
    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    # import ipdb; ipdb.set_trace()
    # exported = cuts.to_shar(
    #     out_shar_dir, fields={"recording": "flac", "target_audio": "flac"}, num_jobs=1, shard_size=shard_size
    # )
    # print(f"...share created")


    # exported = cuts.to_shar(out_shar_dir, fields={"source_audio": "wav", "target_audio": "wav"}, num_jobs=1, shard_size=shard_size)
    # print(f"...share created")
    
    # def filter_cut_with_uneven_toolcalls(cuts):
    #     valid_cuts = []
    #     for cut in cuts:
    #         function_segments = [sup for sup in cut.supervisions[1:] if sup.custom['function'] != '']
    #         if len(function_segments) % 2 == 0:
    #             valid_cuts.append(cut)
    #     return valid_cuts

    # valid_cuts = filter_cut_with_uneven_toolcalls(cuts)
    valid_cuts_list = []
    # ipdb.set_trace()
    for cut in cuts:
        try:
            if cut.load_audio() is not None:
                valid_cuts_list.append(cut)
        except:
            continue
    valid_cuts = CutSet.from_cuts(valid_cuts_list)
    # valid_cuts = CutSet.from_cuts(
    #     [cut for cut in cuts if cut.load_audio() is not None]
    #     )
    print(f"num cuts: {len(cuts)}")
    print(f"num cuts: {len(valid_cuts)}")

    exported = valid_cuts.to_shar(out_shar_dir, fields={}, num_jobs=1, shard_size=shard_size)
    print(f"...share created")
    
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
