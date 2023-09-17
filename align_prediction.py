import utils 
import json 
import copy


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

def main():
    data = utils.load_annotation("../submission/predictions_oldSLU_best_score.jsonl")
    new_data, all_change = fix_cmd(copy.deepcopy(data))
    new_data = fix_miss_field(copy.deepcopy(new_data))
    import json
    with open("./predictions.jsonl", "w", encoding="utf-8") as f:
        for line in new_data:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    main()