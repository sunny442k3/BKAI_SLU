import torch
from transformers import RobertaModel

class BertSLU(torch.nn.Module):

    def __init__(self, n_intent_classes=15, n_token_classes=9, load_pretrained=None):
        super(BertSLU, self).__init__()
        model = RobertaModel.from_pretrained("vinai/phobert-base" if load_pretrained is None else load_pretrained)
        self.model = model
        self.n_intent_classes = n_intent_classes
        self.n_token_classes = n_token_classes
        self.token_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, n_token_classes, bias=True)
        )
        self.intent_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, n_intent_classes, bias=True)
        )
    
    def forward(self, inputs):
        h = self.model(**inputs).last_hidden_state 
        token_out = self.token_classifier(h)
        intent_out = self.intent_classifier(h[:, -1, :])
        return token_out, intent_out 