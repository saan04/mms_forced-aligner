import os
import torch
from pathlib import Path

# Clone uroman and install requirements
!git clone https://github.com/isi-nlp/uroman.git
%cd /kaggle/working/uroman
!pip install -r requirements.txt
!pip install ffmpeg

# Write align.py
%%writefile align.py
import os
import torch
import torchaudio
import sox
import json
import argparse

import os
import re


colon = ":"
comma = ","
exclamation_mark = "!"
period = re.escape(".")
question_mark = re.escape("?")
semicolon = ";"

left_curly_bracket = "{"
right_curly_bracket = "}"
quotation_mark = '"'

basic_punc = (
    period
    + question_mark
    + comma
    + colon
    + exclamation_mark
    + left_curly_bracket
    + right_curly_bracket
)

# General punc unicode block (0x2000-0x206F)
zero_width_space = r"\u200B"
zero_width_nonjoiner = r"\u200C"
left_to_right_mark = r"\u200E"
right_to_left_mark = r"\u200F"
left_to_right_embedding = r"\u202A"
pop_directional_formatting = r"\u202C"

# Here are some commonly ill-typed versions of apostrophe
right_single_quotation_mark = r"\u2019"
left_single_quotation_mark = r"\u2018"

# Language specific definitions
# Spanish
inverted_exclamation_mark = r"\u00A1"
inverted_question_mark = r"\u00BF"


# Hindi
hindi_danda = u"\u0964"

# Egyptian Arabic
# arabic_percent = r"\u066A"
arabic_comma = r"\u060C"
arabic_question_mark = r"\u061F"
arabic_semicolon = r"\u061B"
arabic_diacritics = r"\u064B-\u0652"


arabic_subscript_alef_and_inverted_damma = r"\u0656-\u0657"


# Chinese
full_stop = r"\u3002"
full_comma = r"\uFF0C"
full_exclamation_mark = r"\uFF01"
full_question_mark = r"\uFF1F"
full_semicolon = r"\uFF1B"
full_colon = r"\uFF1A"
full_parentheses = r"\uFF08\uFF09"
quotation_mark_horizontal = r"\u300C-\u300F"
quotation_mark_vertical = r"\uFF41-\uFF44"
title_marks = r"\u3008-\u300B"
wavy_low_line = r"\uFE4F"
ellipsis = r"\u22EF"
enumeration_comma = r"\u3001"
hyphenation_point = r"\u2027"
forward_slash = r"\uFF0F"
wavy_dash = r"\uFF5E"
box_drawings_light_horizontal = r"\u2500"
fullwidth_low_line = r"\uFF3F"
chinese_punc = (
    full_stop
    + full_comma
    + full_exclamation_mark
    + full_question_mark
    + full_semicolon
    + full_colon
    + full_parentheses
    + quotation_mark_horizontal
    + quotation_mark_vertical
    + title_marks
    + wavy_low_line
    + ellipsis
    + enumeration_comma
    + hyphenation_point
    + forward_slash
    + wavy_dash
    + box_drawings_light_horizontal
    + fullwidth_low_line
)

# Armenian
armenian_apostrophe = r"\u055A"
emphasis_mark = r"\u055B"
exclamation_mark = r"\u055C"
armenian_comma = r"\u055D"
armenian_question_mark = r"\u055E"
abbreviation_mark = r"\u055F"
armenian_full_stop = r"\u0589"
armenian_punc = (
    armenian_apostrophe
    + emphasis_mark
    + exclamation_mark
    + armenian_comma
    + armenian_question_mark
    + abbreviation_mark
    + armenian_full_stop
)

lesser_than_symbol = r"&lt;"
greater_than_symbol = r"&gt;"

lesser_than_sign = r"\u003c"
greater_than_sign = r"\u003e"

nbsp_written_form = r"&nbsp"

# Quotation marks
left_double_quotes = r"\u201c"
right_double_quotes = r"\u201d"
left_double_angle = r"\u00ab"
right_double_angle = r"\u00bb"
left_single_angle = r"\u2039"
right_single_angle = r"\u203a"
low_double_quotes = r"\u201e"
low_single_quotes = r"\u201a"
high_double_quotes = r"\u201f"
high_single_quotes = r"\u201b"

all_punct_quotes = (
    left_double_quotes
    + right_double_quotes
    + left_double_angle
    + right_double_angle
    + left_single_angle
    + right_single_angle
    + low_double_quotes
    + low_single_quotes
    + high_double_quotes
    + high_single_quotes
    + right_single_quotation_mark
    + left_single_quotation_mark
)
mapping_quotes = (
    "["
    + high_single_quotes
    + right_single_quotation_mark
    + left_single_quotation_mark
    + "]"
)


# Digits

english_digits = r"\u0030-\u0039"
bengali_digits = r"\u09e6-\u09ef"
khmer_digits = r"\u17e0-\u17e9"
devanagari_digits = r"\u0966-\u096f"
oriya_digits = r"\u0b66-\u0b6f"
extended_arabic_indic_digits = r"\u06f0-\u06f9"
kayah_li_digits = r"\ua900-\ua909"
fullwidth_digits = r"\uff10-\uff19"
malayam_digits = r"\u0d66-\u0d6f"
myanmar_digits = r"\u1040-\u1049"
roman_numeral = r"\u2170-\u2179"
nominal_digit_shapes = r"\u206f"

# Load punctuations from MMS-lab data #INSERT THE FILE PATH
with open("/kaggle/working/punctuations.lst", "r") as punc_f:
    punc_list = punc_f.readlines()

punct_pattern = r""
for punc in punc_list:
    # the first character in the tab separated line is the punc to be removed
    punct_pattern += re.escape(punc.split("\t")[0])

shared_digits = (
    english_digits
    + bengali_digits
    + khmer_digits
    + devanagari_digits
    + oriya_digits
    + extended_arabic_indic_digits
    + kayah_li_digits
    + fullwidth_digits
    + malayam_digits
    + myanmar_digits
    + roman_numeral
    + nominal_digit_shapes
)

shared_punc_list = (
    basic_punc
    + all_punct_quotes
    + greater_than_sign
    + lesser_than_sign
    + inverted_question_mark
    + full_stop
    + semicolon
    + armenian_punc
    + inverted_exclamation_mark
    + arabic_comma
    + enumeration_comma
    + hindi_danda
    + quotation_mark
    + arabic_semicolon
    + arabic_question_mark
    + chinese_punc
    + punct_pattern

)

shared_mappping = {
    lesser_than_symbol: "",
    greater_than_symbol: "",
    nbsp_written_form: "",
    r"(\S+)" + mapping_quotes + r"(\S+)": r"\1'\2",
}

shared_deletion_list = (
    left_to_right_mark
    + zero_width_nonjoiner
    + arabic_subscript_alef_and_inverted_damma
    + zero_width_space
    + arabic_diacritics
    + pop_directional_formatting
    + right_to_left_mark
    + left_to_right_embedding
)

norm_config = {
    "*": {
        "lower_case": True,
        "punc_set": shared_punc_list,
        "del_set": shared_deletion_list,
        "mapping": shared_mappping,
        "digit_set": shared_digits,
        "unicode_norm": "NFKC",
        "rm_diacritics" : False,
    }
}

#=============== Mongolian ===============#

norm_config["mon"] = norm_config["*"].copy()
# add soft hyphen to punc list to match with fleurs
norm_config["mon"]["del_set"] += r"\u00AD"

norm_config["khk"] = norm_config["mon"].copy()

#=============== Hebrew ===============#

norm_config["heb"] = norm_config["*"].copy()
# add "HEBREW POINT" symbols to match with fleurs
norm_config["heb"]["del_set"] += r"\u05B0-\u05BF\u05C0-\u05CF"

#=============== Thai ===============#

norm_config["tha"] = norm_config["*"].copy()
# add "Zero width joiner" symbols to match with fleurs
norm_config["tha"]["punc_set"] += r"\u200D"

#=============== Arabic ===============#
norm_config["ara"] = norm_config["*"].copy()
norm_config["ara"]["mapping"]["ٱ"] = "ا"
norm_config["arb"] = norm_config["ara"].copy()

#=============== Javanese ===============#
norm_config["jav"] = norm_config["*"].copy()
norm_config["jav"]["rm_diacritics"] = True

import json
import re
import unicodedata

def text_normalize(text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False):

    """Given a text, normalize it by changing to lower case, removing punctuations, removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code :
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in ["lower_case", "punc_set","del_set", "mapping", "digit_set", "unicode_norm"]:
        if field not in config:
            config[field] = norm_config["*"][field]


    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += "]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + "]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases  a)text starts with a number b) a number is present somewhere in the middle of the text c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:

        digits_pattern = "[" + config["digit_set"]

        digits_pattern += "]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + "(?=\s)|(?<=\s)"
            + digits_pattern
            + "(?=\s)|(?<=\s)"
            + digits_pattern
            + "$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        from unidecode import unidecode
        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text

import re
import os
import torch
import tempfile
import math
from dataclasses import dataclass
from torchaudio.models import wav2vec2_model

# iso codes with specialized rules in uroman
special_isos_uroman = "ara, bel, bul, deu, ell, eng, fas, grc, ell, eng, heb, kaz, kir, lav, lit, mkd, mkd2, oss, pnt, pus, rus, srp, srp2, tur, uig, ukr, yid".split(",")
special_isos_uroman = [i.strip() for i in special_isos_uroman]

def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def get_uroman_tokens(norm_transcripts, uroman_root_dir, iso = None):
    tf = tempfile.NamedTemporaryFile()
    tf2 = tempfile.NamedTemporaryFile()
    with open(tf.name, "w") as f:
        for t in norm_transcripts:
            f.write(t + "\n")

    assert os.path.exists(f"{uroman_root_dir}/uroman.pl"), "uroman not found"
    cmd = f"perl {uroman_root_dir}/uroman.pl"
    if iso in special_isos_uroman:
        cmd += f" -l {iso} "
    cmd +=  f" < {tf.name} > {tf2.name}"
    os.system(cmd)
    outtexts = []
    with open(tf2.name) as f:
        for line in f:
            line = " ".join(line.strip())
            line =  re.sub(r"\s+", " ", line).strip()
            outtexts.append(line)
    assert len(outtexts) == len(norm_transcripts)
    uromans = []
    for ot in outtexts:
        uromans.append(normalize_uroman(ot))
    return uromans



@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)



def load_model_dict():
    model_path_name = "/tmp/ctc_alignment_mling_uroman_model.pt"

    print("Downloading model and dictionary...")
    if os.path.exists(model_path_name):
        print("Model path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
            model_path_name,
        )
        assert os.path.exists(model_path_name)
    state_dict = torch.load(model_path_name, map_location="cpu")

    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        aux_num_out=31,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dict_path_name = "/tmp/ctc_alignment_mling_uroman_model.dict"
    if os.path.exists(dict_path_name):
        print("Dictionary path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt",
            dict_path_name,
        )
        assert os.path.exists(dict_path_name)
    dictionary = {}
    with open(dict_path_name) as f:
        dictionary = {l.strip(): i for i, l in enumerate(f.readlines())}

    return model, dictionary

def get_spans(tokens, segments):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    sil = "<blank>"
    for (seg_idx, seg) in enumerate(segments):
        if(tokens_idx == len(tokens)):
           assert(seg_idx == len(segments) - 1)
           assert(seg.label == '<blank>')
           continue
        cur_token = tokens[tokens_idx].split(' ')
        ltr = cur_token[ltr_idx]
        if seg.label == "<blank>": continue
        assert(seg.label == ltr)
        if(ltr_idx) == 0: start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                    intervals.append((seg_idx, seg_idx))
                    tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for (idx, (start, end)) in enumerate(intervals):
        span = segments[start:end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == sil:
                pad_start = prev_seg.start if (idx == 0) else int((prev_seg.start + prev_seg.end)/2)
                span = [Segment(sil, pad_start, span[0].start)] + span
        if end+1 < len(segments):
            next_seg = segments[end+1]
            if next_seg.label == sil:
                pad_end = next_seg.end if (idx == len(intervals) - 1) else math.floor((next_seg.start + next_seg.end) / 2)
                span = span + [Segment(sil, span[-1].end, pad_end)]
        spans.append(span)
    return spans

import torchaudio.functional as F

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_emissions(model, audio_file):
    waveform, _ = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)

    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, stride


def get_alignments(
    audio_file,
    tokens,
    model,
    dictionary,
    use_star,
):
    # Generate emissions
    emissions, stride = generate_emissions(model, audio_file)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]

    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)

    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    path, _ = F.forced_align(
        emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank
    )
    path = path.squeeze().to("cpu").tolist()

    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride


def main(args):
    assert not os.path.exists(
        args.outdir
    ), f"Error: Output path exists already {args.outdir}"

    transcripts = []
    with open(args.text_filepath) as f:
        transcripts = [line.strip() for line in f]
    print("Read {} lines from {}".format(len(transcripts), args.text_filepath))

    norm_transcripts = [text_normalize(line.strip(), args.lang) for line in transcripts]
    tokens = get_uroman_tokens(norm_transcripts, args.uroman_path, args.lang)

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts

    segments, stride = get_alignments(
        args.audio_filepath,
        tokens,
        model,
        dictionary,
        args.use_star,
    )
    # Get spans of each line in input text file
    spans = get_spans(tokens, segments)

    os.makedirs(args.outdir)
    with open( f"{args.outdir}/manifest.json", "w") as f:
        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end

            output_file = f"{args.outdir}/segment{i}.flac"

            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000

            tfm = sox.Transformer()
            tfm.trim(audio_start_sec , audio_end_sec)
            tfm.build_file(args.audio_filepath, output_file)

            sample = {
                "audio_start_sec": audio_start_sec,
                "audio_filepath": str(output_file),
                "duration": audio_end_sec - audio_start_sec,
                "text": t,
                "normalized_text":norm_transcripts[i],
                "uroman_tokens": tokens[i],
            }
            f.write(json.dumps(sample) + "\n")

    return segments, stride


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and segment long audio files")
    parser.add_argument(
        "-a", "--audio_filepath", type=str, help="Path to input audio file"
    )
    parser.add_argument(
        "-t", "--text_filepath", type=str, help="Path to input text file "
    )
    parser.add_argument(
        "-l", "--lang", type=str, default="eng", help="ISO code of the language"
    )
    parser.add_argument(
        "-u", "--uroman_path", type=str, default="eng", help="Location to uroman/bin"
    )
    parser.add_argument(
        "-s",
        "--use_star",
        action="store_true",
        help="Use star at the start of transcript",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Output directory to store segmented audio files",
    )
    print("Using torch version:", torch.__version__)
    print("Using torchaudio version:", torchaudio.__version__)
    print("Using device: ", DEVICE)
    args = parser.parse_args()
    main(args)

# Write punctuations.lst
%%writefile punctuations.lst
	7355	INVALID UNICODE	0x81
	5265	INVALID UNICODE	0x90
	75	INVALID UNICODE	0x8
	31	INVALID UNICODE	0x8d
	3	INVALID UNICODE	0x94
	2	INVALID UNICODE	0x8f
	2	INVALID UNICODE	0x1a
	1	INVALID UNICODE	0x9d
	1	INVALID UNICODE	0x93
	1	INVALID UNICODE	0x92
	8647	INVALID UNICODE	0xe295
	6650	INVALID UNICODE	0xf21d
	6234	INVALID UNICODE	0xf62d
	4815	INVALID UNICODE	0xf173
	4789	INVALID UNICODE	0xe514
	4409	INVALID UNICODE	0xe293
	3881	INVALID UNICODE	0xf523
	3788	INVALID UNICODE	0xe233
	2448	INVALID UNICODE	0xf50f
	2177	INVALID UNICODE	0xe232
	1955	INVALID UNICODE	0xea7b
	1926	INVALID UNICODE	0xf172
	973	INVALID UNICODE	0xe290
	972	INVALID UNICODE	0xf519
	661	INVALID UNICODE	0xe292
	591	INVALID UNICODE	0xe328
	509	INVALID UNICODE	0xe2fa
	458	INVALID UNICODE	0xe234
	446	INVALID UNICODE	0xe043
	419	INVALID UNICODE	0xe040
	399	INVALID UNICODE	0xe2fb
	387	INVALID UNICODE	0xe32b
	381	INVALID UNICODE	0xe236
	374	INVALID UNICODE	0xf511
	314	INVALID UNICODE	0xe517
	296	INVALID UNICODE	0xe2fe
	293	INVALID UNICODE	0xe492
	291	INVALID UNICODE	0xf52d
	289	INVALID UNICODE	0xe2fc
	195	INVALID UNICODE	0xf521
	190	INVALID UNICODE	0xe516
	182	INVALID UNICODE	0xe041
	178	INVALID UNICODE	0xf529
	113	INVALID UNICODE	0xe2f9
	87	INVALID UNICODE	0xe2d9
	78	INVALID UNICODE	0xe32a
	76	INVALID UNICODE	0xe291
	74	INVALID UNICODE	0xe296
	66	INVALID UNICODE	0xe518
	52	INVALID UNICODE	0xe32c
	46	INVALID UNICODE	0xe2db
	41	INVALID UNICODE	0xe231
	34	INVALID UNICODE	0xf522
	33	INVALID UNICODE	0xf518
	32	INVALID UNICODE	0xf513
	27	INVALID UNICODE	0xe32d
	25	INVALID UNICODE	0xe32e
	23	INVALID UNICODE	0xe06b
	15	INVALID UNICODE	0xea01
	12	INVALID UNICODE	0xe294
	11	INVALID UNICODE	0xe203
	8	INVALID UNICODE	0xf218
	7	INVALID UNICODE	0xe070
	7	INVALID UNICODE	0xe013
	5	INVALID UNICODE	0xe2de
	4	INVALID UNICODE	0xe493
	3	INVALID UNICODE	0xf7e8
	3	INVALID UNICODE	0xf7d0
	3	INVALID UNICODE	0xe313
	2	INVALID UNICODE	0xe329
	2	INVALID UNICODE	0xe06d
	2	INVALID UNICODE	0xe003
	1	INVALID UNICODE	0xf50e
	1	INVALID UNICODE	0xf171
	1	INVALID UNICODE	0xe01d
⁯	71	NOMINAL DIGIT SHAPES	0x206f
⁠	3	WORD JOINER	0x2060
―	126545	HORIZONTAL BAR	0x2015
־	1028	HEBREW PUNCTUATION MAQAF	0x5be
)	98429	RIGHT PARENTHESIS	0x29
]	27108	RIGHT SQUARE BRACKET	0x5d
⌋	1567	RIGHT FLOOR	0x230b
〕	97	RIGHT TORTOISE SHELL BRACKET	0x3015
】	36	RIGHT BLACK LENTICULAR BRACKET	0x3011
﴾	14	ORNATE LEFT PARENTHESIS	0xfd3e
&	170517	AMPERSAND	0x26
།	106330	TIBETAN MARK SHAD	0xf0d
።	90203	ETHIOPIC FULL STOP	0x1362
፥	60484	ETHIOPIC COLON	0x1365
༌	60464	TIBETAN MARK DELIMITER TSHEG BSTAR	0xf0c
။	51567	MYANMAR SIGN SECTION	0x104b
/	46929	SOLIDUS	0x2f
၊	38042	MYANMAR SIGN LITTLE SECTION	0x104a
·	37985	MIDDLE DOT	0xb7
‸	36310	CARET	0x2038
*	34793	ASTERISK	0x2a
۔	32432	ARABIC FULL STOP	0x6d4
፤	31906	ETHIOPIC SEMICOLON	0x1364
၏	21519	MYANMAR SYMBOL GENITIVE	0x104f
។	20834	KHMER SIGN KHAN	0x17d4
꓾	15773	LISU PUNCTUATION COMMA	0xa4fe
᙮	13473	CANADIAN SYLLABICS FULL STOP	0x166e
꤯	12892	KAYAH LI SIGN SHYA	0xa92f
⵰	11478	TIFINAGH SEPARATOR MARK	0x2d70
꓿	11118	LISU PUNCTUATION FULL STOP	0xa4ff
॥	10763	DEVANAGARI DOUBLE DANDA	0x965
؞	10403	ARABIC TRIPLE DOT PUNCTUATION MARK	0x61e
၍	8936	MYANMAR SYMBOL COMPLETED	0x104d
·	8431	GREEK ANO TELEIA	0x387
†	7477	DAGGER	0x2020
၌	6632	MYANMAR SYMBOL LOCATIVE	0x104c
፣	5719	ETHIOPIC COMMA	0x1363
៖	5528	KHMER SIGN CAMNUC PII KUUH	0x17d6
꤮	4791	KAYAH LI SIGN CWI	0xa92e
※	3439	REFERENCE MARK	0x203b
፦	2727	ETHIOPIC PREFACE COLON	0x1366
•	1749	BULLET	0x2022
¶	1507	PILCROW SIGN	0xb6
၎	1386	MYANMAR SYMBOL AFOREMENTIONED	0x104e
﹖	1224	SMALL QUESTION MARK	0xfe56
;	975	GREEK QUESTION MARK	0x37e
…	827	HORIZONTAL ELLIPSIS	0x2026
%	617	PERCENT SIGN	0x25
・	468	KATAKANA MIDDLE DOT	0x30fb
༎	306	TIBETAN MARK NYIS SHAD	0xf0e
‡	140	DOUBLE DAGGER	0x2021
#	137	NUMBER SIGN	0x23
@	125	COMMERCIAL AT	0x40
፡	121	ETHIOPIC WORDSPACE	0x1361
៚	55	KHMER SIGN KOOMUUT	0x17da
៕	49	KHMER SIGN BARIYOOSAN	0x17d5
﹐	10	SMALL COMMA	0xfe50
༅	6	TIBETAN MARK CLOSING YIG MGO SGAB MA	0xf05
༄	6	TIBETAN MARK INITIAL YIG MGO MDUN MA	0xf04
．	2	FULLWIDTH FULL STOP	0xff0e
﹗	2	SMALL EXCLAMATION MARK	0xfe57
﹕	2	SMALL COLON	0xfe55
‰	2	PER MILLE SIGN	0x2030
･	1	HALFWIDTH KATAKANA MIDDLE DOT	0xff65
(	98504	LEFT PARENTHESIS	0x28
[	27245	LEFT SQUARE BRACKET	0x5b
⌊	1567	LEFT FLOOR	0x230a
〔	95	LEFT TORTOISE SHELL BRACKET	0x3014
【	36	LEFT BLACK LENTICULAR BRACKET	0x3010
﴿	14	ORNATE RIGHT PARENTHESIS	0xfd3f
_	4851	LOW LINE	0x5f
$	72	DOLLAR SIGN	0x24
€	14	EURO SIGN	0x20ac
£	2	POUND SIGN	0xa3
~	27462	TILDE	0x7e
=	11450	EQUALS SIGN	0x3d
|	8430	VERTICAL LINE	0x7c
−	3971	MINUS SIGN	0x2212
≫	1904	MUCH GREATER-THAN	0x226b
≪	1903	MUCH LESS-THAN	0x226a
+	1450	PLUS SIGN	0x2b
＜	345	FULLWIDTH LESS-THAN SIGN	0xff1c
＞	344	FULLWIDTH GREATER-THAN SIGN	0xff1e
¬	5	NOT SIGN	0xac
×	4	MULTIPLICATION SIGN	0xd7
→	2	RIGHTWARDS ARROW	0x2192
᙭	537	CANADIAN SYLLABICS CHI SIGN	0x166d
°	499	DEGREE SIGN	0xb0
႟	421	MYANMAR SYMBOL SHAN EXCLAMATION	0x109f
�	192	REPLACEMENT CHARACTER	0xfffd
⌟	54	BOTTOM RIGHT CORNER	0x231f
⌞	54	BOTTOM LEFT CORNER	0x231e
©	2	COPYRIGHT SIGN	0xa9
 	40	NARROW NO-BREAK SPACE	0x202f
 	1	SIX-PER-EM SPACE	0x2006
˜	40261	SMALL TILDE	0x2dc
^	6469	CIRCUMFLEX ACCENT	0x5e
¯	20	MACRON	0xaf
ˇ	191442	CARON	0x2c7
ⁿ	38144	SUPERSCRIPT LATIN SMALL LETTER N	0x207f
ـ	9440	ARABIC TATWEEL	0x640
ๆ	6766	THAI CHARACTER MAIYAMOK	0xe46
ៗ	3310	KHMER SIGN LEK TOO	0x17d7
々	678	IDEOGRAPHIC ITERATION MARK	0x3005
ໆ	430	LAO KO LA	0xec6
ー	319	KATAKANA-HIRAGANA PROLONGED SOUND MARK	0x30fc
ⁱ	137	SUPERSCRIPT LATIN SMALL LETTER I	0x2071
৷	11056	BENGALI CURRENCY NUMERATOR FOUR	0x9f7
⅓	26	VULGAR FRACTION ONE THIRD	0x2153
½	26	VULGAR FRACTION ONE HALF	0xbd
¼	4	VULGAR FRACTION ONE QUARTER	0xbc
⅟	1	FRACTION NUMERATOR ONE	0x215f
⁄	57	FRACTION SLASH	0x2044

%cd /kaggle/working/uroman
# give the paths as required
audio_dir = 
text_dir = 
uroman_path = 
output_base_dir = 
output_temp = 

audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

for audio_file in audio_files:
    torch.cuda.empty_cache()
    f = audio_file.split("-")
    name = "mkb-"
    name = name + f[-2].lower() + "-" + f[-1].split(".")[0]
    try:
        text_file = os.path.join(text_dir, f"{name}.txt")
        output_dir = os.path.join(output_temp, name)
        if not os.path.exists(text_file):
            print(f"Text file not found for: {audio_file}, skipping...")
            continue
        
        command = f'python "/kaggle/working/align.py" -a "{os.path.join(audio_dir, audio_file)}" -t "{text_file}" -l mni -u "{uroman_path}" -o "{output_dir}/"'
        os.system(command)
        print(f"Processed: {audio_file} with {text_file}")
        
        with open(f'{output_dir}/manifest.json','r') as f:
            with open(f'{output_base_dir}/{name}.json','w') as new_f:
                data = f.read()
                new_f.write(data)
        print("Manifest.json Saved")
        
        command = f'rm -r {output_dir}'
        os.system(command)
        print("Removed files and folder")
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")

print("Processing complete.")