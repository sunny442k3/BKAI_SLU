
import json
import gdown
# import kenlm
# from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel


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

# def get_decoder_ngram_model(tokenizer, ngram_lm_path):
#     vocab_dict = tokenizer.get_vocab()
#     sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
#     vocab = [x[1] for x in sort_vocab][:-2]
#     vocab_list = vocab
#     # convert ctc blank character representation
#     vocab_list[tokenizer.pad_token_id] = ""
#     # replace special characters
#     vocab_list[tokenizer.unk_token_id] = ""
#     # convert space character representation
#     vocab_list[tokenizer.word_delimiter_token_id] = " "
#     # specify ctc blank char index, since conventially it is the last entry of the logit matrix
#     alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
#     lm_model = kenlm.Model(ngram_lm_path)
#     decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
#     return decoder

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
