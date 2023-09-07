import torch
from transformers import RobertaModel, Wav2Vec2Model

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
    
class E2ESLU(torch.nn.Module):
    def __init__(
            self,
            n_intent_classes,
            n_token_classes,
            n_logit_classes,
            pretrained_w2v, 
            pretrained_phobert=None
    ):
        super(E2ESLU, self).__init__()
        self.n_intent_classes = n_intent_classes
        self.n_token_classes = n_token_classes
        self.n_logit_classes = n_logit_classes
        self.model1 = Wav2Vec2Model.from_pretrained(pretrained_w2v)
        for p in self.model1.parameters():
            p.requires_grad = False

        self.proj = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 768)
        )

        self.model2 = RobertaModel.from_pretrained(pretrained_phobert if pretrained_phobert is not None else "vinai/phobert-base-v2")

        self.token_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, self.n_token_classes, bias=True)
        )
        self.intent_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, self.n_intent_classes, bias=True)
        )

        self.lm_head = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, self.n_logit_classes, bias=True),
        )
    
    def forward(self, inputs):
        h = self.model1(**inputs).last_hidden_state.clone().detach() 
        h = self.proj(h)
        features = self.model2(inputs_embeds=h[:, :256, :]).last_hidden_state
        token_out = self.token_classifier(features)
        intent_out = self.intent_classifier(features[:, -1, :])
        logits = self.lm_head(features)
        return logits, token_out, intent_out 