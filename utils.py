
import json


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
