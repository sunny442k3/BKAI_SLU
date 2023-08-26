
import os
import re
import json
import soundfile as sf
from torch.utils.data import Dataset

class Wav2VecDataset(Dataset):

    def __init__(self, root_path, files_id, labels=None, load_all=False):
        self.root_path = root_path 
        self.files_id = files_id 
        self.labels = labels 
        self.load_all = load_all
        if load_all:
            self._load_all_data()
    
    def _process_sound_file(self, idx):
        speech, samplerate  = sf.read(os.path.join(self.root_path, self.files_id[idx]))
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        clean_txt = lambda txt: re.sub(chars_to_ignore_regex, '', txt.lower()).lower()
        label = clean_txt(self.labels[idx]["sentence"]) if self.labels is not None else None
        return {"input_values": speech, "sample_rate": samplerate, "label": label, "file": self.files_id[idx]}
    
    def _load_all_data(self):
        self.all_data = [
            self._process_sound_file(i) for i in range(len(self.files_id))
        ]
        
    def __len__(self):
        return len(self.files_id)

    def __getitem__(self, idx):
        if self.load_all:
            return self.all_data[idx]
        data = self._process_sound_file(idx)
        return data

class BertDataset(Dataset):
    def __init__(self, all_text, labels=None, map_intent=None):
        self.all_text = all_text
        self.labels = labels # [Token cls, Intent cls]
        self.map_intent = map_intent

    def _process_data(self, idx):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\'\"]'
        clean_txt = lambda txt: re.sub(chars_to_ignore_regex, '', txt.lower()).lower().replace("["," [ ").replace("]", " ] ").replace("  ", " ")
        text = clean_txt(self.all_text[idx])
        # label = clean_txt(self.annotations[idx]["sentence_annotation"])
        if self.labels is None or self.map_intent is None:
            return text, None, None
        token_label = [0] + self.labels[0][idx] + [0]
        intent_label = [self.map_intent[self.labels[1][idx]]]
        return text, token_label, intent_label

    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, idx):
        text, token_label, intent_label = self._process_data(idx)
        return {"text": text, "token_label": token_label, "intent_label": intent_label}    

    