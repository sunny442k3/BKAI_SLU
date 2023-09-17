import json
import utils
import random
import copy as cp
from transformers import AutoTokenizer



def dcp(data):
    return cp.deepcopy(data)

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.close()

def load_json(filename):
    return json.load(open(filename, "r", encoding="utf-8"))


def make_group_by_intent(data):
    groups = {k: [] for k in utils.MAP_INTENT.values()}
    for row in data.values():
        gr = row["intent_label"]
        groups[gr].append(row)
    return groups

def extract_frame(data):
    groups = {k: [] for k in utils.MAP_TOKEN.values()}
    frames = []
    for row in data:
        tmp = [row["token"][0]]
        frame = []
        for i in range(1, len(row["token"])):
            if row["token_label"][i] == row["token_label"][i-1]:
                tmp += [row["token"][i]]
                continue
            if len(tmp):
                frame += [" ".join(tmp)]
            tmp = [row["token"][i]]
        if len(tmp):
            frame += [" ".join(tmp)]
        frame_label = []
        token_label = dcp(row["token_label"])
        for i in frame:
            l = len(i.split(" "))
            frame_label.append(token_label[:l])
            token_label = token_label[l:]
            if frame_label[-1][-1] != 0:
                groups[frame_label[-1][-1]].append(i)
        frames.append({"token": frame, "label": frame_label})
    groups = {k: list(set(v)) for k, v in groups.items()}
    return frames, groups


def data_augment(intent_groups, max_size=1130):
    new_groups = {}
    for k, v in intent_groups.items():
        if len(v) >= max_size:
            new_groups[k] = v
            continue
        frames, groups = extract_frame(dcp(v))
        rem_len = max_size - len(v)
        print(f"[+] Augment in group: {k} - Num sample augment: {rem_len}")
        for _ct in range(rem_len):
            aug_frame_idx = random.randint(0, len(frames)-1)
            aug_frame = frames[aug_frame_idx]
            for idx, tmp in enumerate(aug_frame["label"]):
                if tmp[-1] in [0, 4, 5]:
                    continue
                idx_group = tmp[-1]
                idx_type = random.randint(0, len(groups[idx_group])-1)
                new_type = groups[idx_group][idx_type]
                aug_frame["token"][idx] = new_type
                aug_frame["label"][idx] = [idx_group for jj in range(len(new_type.split(" ")))]
            frames.append(aug_frame)
            random.shuffle(frames)
            print("\r", end="")
            print(f"\r {_ct+1} / {rem_len}", end = "" if _ct != rem_len-1 else "\n")
        new_v = []
        for idx, frame in enumerate(frames):
            tmp = {"names": [], "label": [], "sentence": [], "intent": v[0]["intent"], "intent_label": v[0]["intent_label"]}
            for i in range(len(frame["token"])):
                tmp["sentence"].append(frame["token"][i])
                tmp["label"] += frame["label"][i]
                if frame["label"][i][-1] == 0:
                    continue
                tmp["names"].append(frame["token"][i])
            tmp["sentence"] = " ".join(tmp["sentence"])
            new_v.append(tmp)
            print("\r", end="")
            print(f"\r {idx+1} / {len(frames)}", end = "" if idx != len(frames)-1 else "\n")
        new_groups[k] = new_v
    return new_groups


def main():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    data = load_json("./dataset/train_token_labels_20230909.json")
    for k, v in data.items():
        v["token"] = tokenizer.tokenize(v["sentence"])
        v["token_label"] = v["label"]
        del v["label"]
        data[k] = v

    intent_groups = make_group_by_intent(dcp(data))
    len_groups = [len(v) for k, v in intent_groups.items()]
    for k, v in utils.MAP_INTENT.items():
        print(k, ":", len_groups[v])

    augmented_data =  data_augment(intent_groups)
    list_augmented_data = [i for k, v in augmented_data.items() for i in dcp(v)]

    save_json(dcp(list_augmented_data), "./dataset/augmented_data.json")


if __name__ == "__main__":
    main()
