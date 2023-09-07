import re
import json
import utils
from transformers import AutoTokenizer


def load_notation(path):
    data = open(path, "rb").readlines()
    data = [json.loads(i) for i in data]
    return data 

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.close()

def make_labels(tokenizer, all_data):
    map_tag2ids = utils.MAP_TOKEN
    logs_data = []
    for idx_, data in enumerate(all_data):
        txt = data["label"]
        label = [0 for _ in range(len(data["input"]))]
        current_idx = 0
        fl = 0
        tag = []
        name = []
        tmp_logs = []
        for j, i in enumerate(txt):
            if fl == 0:
                if i == '[':
                    fl = 1
                else:
                    current_idx += 1
                continue 
            if fl == 1:
                if i == ':':
                    word = tokenizer.decode(tokenizer.convert_tokens_to_ids(tag))
                    tag = word
                    fl = 2
                else:
                    tag += [i]
                continue 
            if fl == 2:
                if i == "]":
                    fl = 0
                    name = tokenizer.decode(tokenizer.convert_tokens_to_ids(name))
                    tmp_logs.append(name)
                    name = []
                    tag = []
                else:
                    label[current_idx] = map_tag2ids[tag]
                    current_idx += 1
                    name += [i]
        logs_data.append({
            "names": tmp_logs,
            "label": label
        })
    return logs_data

def main():
    annotations = load_notation("./dataset/train.jsonl")
    chars_to_ignore_regex = '[\,\?\.\!\-\;\'\"]'
    all_data = [{
        "input": re.sub(chars_to_ignore_regex, " ", i["sentence"]).lower().replace("["," [ ").replace("]", " ] ").replace("  ", " "),
        "label": re.sub(chars_to_ignore_regex, " ", i["sentence_annotation"]).lower().replace("["," [ ").replace("]", " ] ").replace("  ", " ")
    } for i in annotations]
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    all_data = [{
        "input": tokenizer.tokenize(i["input"]),
        "label": tokenizer.tokenize(i["label"])
    } for i in all_data]

    logs_data = make_labels(tokenizer, all_data)
    save_json("./dataset/train_token_labels.json", [i["label"] for i in logs_data])
    bk_err = []
    for ct, (data, _label) in enumerate(zip(all_data, logs_data)):
        txt = data["input"]
        label = _label["label"]
        all_words = []
        word = [] if label[0] == 0 else [txt[0]]
        for idx_c, c in enumerate(label[1:], 1):
            if c != 0:
                if c == label[idx_c-1]:
                    word += [txt[idx_c]]  
                    continue  
                if label[idx_c-1] != 0:
                    word = tokenizer.decode(tokenizer.convert_tokens_to_ids(word)) 
                    all_words.append(word)
                word = [txt[idx_c]]
            else:
                if label[idx_c-1] != 0:
                    word = tokenizer.decode(tokenizer.convert_tokens_to_ids(word))
                    all_words.append(word)
                word = []
        if len(word):
            all_words.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(word)))
            word = []
        ce = [i == j for i, j in zip(all_words, _label["names"])]
        ce = sum(ce)
        if ce != len(_label["names"]):
            print(ct)
            bk_err.append(ct)
    if len(bk_err):
        print(f"[!] Found {len(bk_err)} items incorrect with label mask")
        save_json("./dataset/bk_error.json", bk_err)
    else:
        print(f"[+] All label mask are matched with annotation")

if __name__ == "__main__":
    main()