
import json
import gdown
import re

# import kenlm
# from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

def build_ngram_dataset(annotation_file, out_file="./full_text.txt"):
    data = load_annotation(annotation_file)
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    clean_txt = lambda txt: re.sub(chars_to_ignore_regex, '', txt.lower()).lower()
    all_sen = [clean_txt(i["sentence"]) for i in data]
    for i in range(len(all_sen)):
        while "  " in all_sen[i]:
            all_sen[i] = all_sen[i].replace("  ", " ")
        while " %" in all_sen[i]:
            all_sen[i] = all_sen[i].replace(" %", "%")
    with open(out_file, "w") as file:
        file.write(" ".join(all_sen))

def align_ngram_set(ngram_path="./4gram.arpa"):
    with open(ngram_path, "r") as read_file, open("./4gram_correct.arpa", "w") as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count=line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)

def drive_download(idx, output):
    url = 'https://drive.google.com/uc?id=' + idx
    gdown.download(url, output, quiet=False)

def download_data():
    drive_download("1ZBL3h6bHMmd8MIUNXqg72PucUkC9ZSWJ", "./dataset/train_data.zip")
    drive_download("1ZepptsTrVSjQEx-dpBBmQ2b7xYFLn_64", "./dataset/public_test.zip")
    drive_download("1K_07kix1OgBGO2FNPh-Lxqr1yLbtqFYt", "./dataset/train.jsonl")

def load_annotation(path):
    data = open(path, "rb").readlines()
    data = [json.loads(i) for i in data]
    return data


def load_json(path):
    return json.load(open(path, "r"))


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.close()


def weight_decay(model, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

MAP_INTENT = {
    'Giảm độ sáng của thiết bị': 0,
    'Đóng thiết bị': 1,
    'Hủy hoạt cảnh': 2,
    'Tắt thiết bị': 3,
    'Tăng âm lượng của thiết bị': 4,
    'Giảm mức độ của thiết bị': 5,
    'Bật thiết bị': 6,
    'Tăng mức độ của thiết bị': 7,
    'Tăng nhiệt độ của thiết bị': 8,
    'Kiểm tra tình trạng thiết bị': 9,
    'Mở thiết bị': 10,
    'Giảm âm lượng của thiết bị': 11,
    'Kích hoạt cảnh': 12,
    'Giảm nhiệt độ của thiết bị': 13,
    'Tăng độ sáng của thiết bị': 14
}

MAP_TOKEN = {
    "word": 0,
    "time at": 1,
    "device": 2,
    "changing value": 3,
    "scene": 4,
    "command": 5,
    "location": 6,
    "duration": 7,
    "target number": 8
}
