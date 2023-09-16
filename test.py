# %%

import utils 
import json 
import copy
# %%
test_data = json.load(open("./4gram_test_sentences_v3_32w.json", "r"))
# %%
def fix_type1(data, intent, word):
    change = []
    new_data = []
    for i in range(len(data)):
        record = copy.deepcopy(data[i])
        # record = {k: v for k, v in data[i].items()}
        if record["intent"] != intent:
            new_data.append(record) 
            continue 
        fc = lambda x: x["type"] == "command" and x["filler"] == word
        g = [fc(j) for j in record["entities"]]
        new_entities = copy.deepcopy(record["entities"])
        if sum(g) == 0:
            change.append(record["file"])
            new_entities += [{"type": "command", "filler": word}]
        record["entities"] = copy.deepcopy(new_entities)
        new_data.append(record)

    return new_data, change

def fix_type2(data, intent, word):
    change = []
    new_data = []
    for i in range(len(data)):
        record = copy.deepcopy(data[i])
        # record = {k: v for k, v in data[i].items()}
        if record["intent"] != intent:
            new_data.append(record) 
            continue 
        new_entities = copy.deepcopy(record["entities"])
        check_scene = [j for j in new_entities if j["type"] == "scene" and len(j["filler"]) != 0]
        if len(check_scene) == 0:
            new_entities += [{"type":"scene", "filler": test_data[record["file"]]}]

        record["entities"] = copy.deepcopy(new_entities)
        new_data.append(record)
    return new_data, change

def fix_miss_field(data):
    new_data = []
    for i in range(len(data)):
        record = copy.deepcopy(data[i])
        new_entities = []
        for en in copy.deepcopy(record["entities"]):
            if len(en["filler"].strip()) == 0:
                continue 
            new_entities.append(copy.deepcopy(en))
        record["entities"] = copy.deepcopy(new_entities)
        new_data.append(record)
    return new_data
        
# %%
def fix_cmd(data):
    map_fix = [
    ["Kiểm tra tình trạng thiết bị", "kiểm tra"],
    ["Giảm âm lượng của thiết bị", "giảm"],
    ["Tắt thiết bị", "tắt"],
    ["Giảm nhiệt độ của thiết bị", "giảm"],
    [ "Đóng thiết bị", "đóng" ],
    [ "Tăng mức độ của thiết bị", "tăng" ],
    [ "Bật thiết bị", "bật" ],
    [ "Tăng nhiệt độ của thiết bị", "tăng" ],
    [ "Tăng âm lượng của thiết bị", "tăng" ],
    [ "Tăng độ sáng của thiết bị", "tăng" ],
    [ "Giảm độ sáng của thiết bị", "giảm" ],
    [ "Mở thiết bị", "mở" ],
    [ "Giảm mức độ của thiết bị", "giảm" ]
]
    new_data = copy.deepcopy(data)
    all_change = []
    for i in map_fix:
        res_data, change = fix_type1(
            copy.deepcopy(new_data),
            i[0], i[1]
        )
        new_data = copy.deepcopy(res_data)
        print(i[0], len(change))
        all_change += change
    return new_data, all_change

# %%
def fix_scene(data):
    map_fix = [
        [  "Hủy hoạt cảnh",  "scene" ],
        [ "Kích hoạt cảnh", "scene" ],
    ]

    new_data = copy.deepcopy(data)
    for i in map_fix:
        res_data, change = fix_type2(
            copy.deepcopy(new_data),
            i[0], i[1]
        )
        new_data = copy.deepcopy(res_data)
        print(i[0], len(change))
    return new_data



# %%
data = utils.load_annotation("./predictions.jsonl")
data[:2]

# %%
new_data, all_change = fix_cmd(copy.deepcopy(data))

# %%
all_change

# %%
new_data = fix_miss_field(copy.deepcopy(new_data))

# %%
tmp = [(i, test_data[i["file"]]) for i in data if i["file"] in all_change]

# %%

for i in tmp[10:]:
    print("="*20)
    print(i[0]["intent"], "\n", i[1])
    for j in i[0]["entities"]:
        print(j)
    print("="*20)
# %%
import json

with open("./predictions_new.jsonl", "w", encoding="utf-8") as f:
    for line in new_data:
        json.dump(line, f, ensure_ascii=False)
        f.write('\n')
# %%
