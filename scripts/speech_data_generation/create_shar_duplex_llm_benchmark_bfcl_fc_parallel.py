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
                speaker=convs[0]["from"],
                language="EN",
            ),
        )

        # total_duration += user_duration 
                    # ipdb.set_trace()
        # if user_path != '':
        sample_rate = Recording.from_file(user_path).sampling_rate
        try:
            cur_user_audio = Recording.from_file(user_path).resample(sample_rate).load_audio() # ignore metadata in audio
        except:
            ipdb.set_trace()
        # cur_user_agent_added = False
        # elif cur_user_agent_added:
        #     cur_user_audio = None
        
        if user_duration > 0:
            total_duration += (user_duration +turn_silence_sec)

        
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
        assistant_duration = 0

        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=total_duration,
                duration=assistant_duration, #cut.recording.duration,
                # text=convs[i + 1]["transcript"],
                text=assistant_transcript,
                custom={'function': assistant_function},
                speaker="assistant",
                language="EN",
            ),
        )

        total_duration += assistant_duration

        # ipdb.set_trace()
        # if assistant_path != '':
        #     # agent_duration = Recording.from_file(assistant_path).duration
        #     cur_agent_audio = Recording.from_file(assistant_path).load_audio()
        #     cur_user_agent_added = False
        # elif cur_user_agent_added:
        #     cur_agent_audio = None
        cur_agent_audio = np.zeros_like(cur_user_audio)

        if cur_user_audio is not None and cur_agent_audio is not None and not cur_user_agent_added:
            # print(cut.id)
            if i != 0: # overlap this turn with previos one
                if overlap_samples > 0:
                    user_audio = user_audio[:, :-overlap_samples]
                total_dur -= overlap_sec
            user_audio = np.concatenate([user_audio, cur_user_audio, silence_padding, 0 * cur_agent_audio], axis=1)
            user_duration += turn_silence_sec
            # total_duration += turn_silence_sec

            if i != 0:
                if overlap_samples > 0:
                    agent_audio = np.concatenate([agent_audio, 0 * cur_user_audio[:, :-overlap_samples], silence_padding, cur_agent_audio], axis=1)
                else:
                    agent_audio = np.concatenate([agent_audio, 0 * cur_user_audio, silence_padding, cur_agent_audio], axis=1)
            else:
                agent_audio = np.concatenate([agent_audio, 0 * cur_user_audio, silence_padding, cur_agent_audio], axis=1)

            # total_duration += assistant_duration
            # total_duration += user_duration + agent_duration

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
        cut.duration = total_duration
        cut.start = 0.0
        try:
            assert user_audio.shape == agent_audio.shape
        except:
            # ipdb.set_trace()
            # assert user_audio.shape < agent_audio.shape
            unequal += 1
            diff = agent_audio.shape[1] - user_audio.shape[1]
            if diff > 0:
                user_audio = np.concatenate([user_audio, np.zeros((1, int(diff)))], axis=1) # add silence
            else:
                agent_audio = np.concatenate([agent_audio, np.zeros((1, int(np.abs(diff))))], axis=1) # add silence
            # ipdb.set_trace()
            assert user_audio.shape == agent_audio.shape

        save_audio(f"/tmp/u{j}1.wav", user_audio, sample_rate)
        cut.recording = Recording.from_file(f"/tmp/u{j}1.wav")
        save_audio(f"/tmp/u{j}2.wav", agent_audio, sample_rate)
        cut.target_audio = Recording.from_file(f"/tmp/u{j}2.wav")
        # if cut.id == "glaive-functioncalling-v2+toolcall+respond_synthesized_dial_4_turn_3_Assistant_audio-4":
        #     import ipdb; ipdb.set_trace()

        ## method 2
        # user_stream = BytesIO()
        # agent_stream = BytesIO()
        # save_audio(dest=user_stream, src=user_audio, sampling_rate=sample_rate, format="wav")
        # save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sample_rate, format="wav")

        # user_stream.seek(0)
        # agent_stream.seek(0)
        # # # import ipdb; ipdb.set_trace()
        # cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut.id}_user")
        # cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut.id}_agent")

        ## method 3:
        # Save the in-memory audio to temporary files
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_user_file:
        #     temp_user_file.write(user_stream.getvalue())
        #     temp_user_path = temp_user_file.name

        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_agent_file:
        #     temp_agent_file.write(agent_stream.getvalue())
        #     temp_agent_path = temp_agent_file.name
        # cut.recording = Recording.from_file(temp_user_path, f"{cut.id}_user")
        # cut.target_audio = Recording.from_file(temp_agent_path, f"{cut.id}_agent")

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
