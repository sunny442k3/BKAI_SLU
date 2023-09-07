
import os
import re
import json
import soundfile as sf
from torch.utils.data import Dataset
import glob

class Wav2VecDataset(Dataset):

    def __init__(self, root_path, files_id, labels=None):
        self.root_path = root_path 
        self.files_id = files_id 
        self.labels = labels 
    
    def _process_sound_file(self, idx):
        speech, samplerate  = sf.read(os.path.join(self.root_path, self.files_id[idx]))
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        clean_txt = lambda txt: re.sub(chars_to_ignore_regex, '', txt.lower()).lower()
        label = clean_txt(self.labels[idx]["sentence"]) if self.labels is not None else None
        return {"input_values": speech, "sample_rate": samplerate, "label": label, "file": self.files_id[idx]}
        
    def __len__(self):
        return len(self.files_id)

    def __getitem__(self, idx):
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


class E2EDataset(Dataset):
    def __init__(self, root_path, files_id, annotations=None, labels=None, map_intent=None):
        self.root_path = root_path 
        self.files_id = files_id 
        self.labels = labels 
        self.annotations = annotations
        self.map_intent = map_intent

    def _process_data(self, idx):
        if self.annotations is None:
            return None, None, None
        chars_to_ignore_regex = '[\,\?\.\!\-\;\'\"]'
        clean_txt = lambda txt: re.sub(chars_to_ignore_regex, '', txt.lower()).lower().replace("["," [ ").replace("]", " ] ").replace("  ", " ")
        text = clean_txt(self.annotations[idx]["sentence"])
        if self.labels is None or self.map_intent is None:
            return text, None, None
        token_label = [0] + self.labels[0][idx] + [0]
        intent_label = [self.map_intent[self.labels[1][idx]]]
        return text, token_label, intent_label
    
    def _process_sound_file(self, idx):
        speech, samplerate  = sf.read(os.path.join(self.root_path, self.files_id[idx]))
        return {"input_values": speech, "sample_rate": samplerate, "file": self.files_id[idx]}
        
    def __len__(self):
        return len(self.files_id)

    def __getitem__(self, idx):
        data = self._process_sound_file(idx)
        text, token_label, intent_label = self._process_data(idx)
        data["sentence"] = text 
        data["token_label"] = token_label
        data["intent_label"] = intent_label
        return data
        # return {"text": text, "token_label": token_label, "intent_label": intent_label}