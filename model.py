import torch
import torch.nn as nn
from transformers import RobertaModel, Wav2Vec2ForCTC
import transformers.models.wav2vec2.modeling_wav2vec2 as lib

class BertSLU(torch.nn.Module):

    def __init__(self, n_intent_classes=15, n_token_classes=9, load_pretrained=None):
        super(BertSLU, self).__init__()
        model = RobertaModel.from_pretrained("vinai/phobert-base-v2" if load_pretrained is None else load_pretrained)
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
    
    
class GICLayer(nn.Module):
    def __init__(self, vocab_size=111, embedding_dim=1024):
        super(GICLayer, self).__init__()
        self.vocab_size = vocab_size 
        self.embedding_dim = embedding_dim
        self.classify_head = torch.nn.Linear(embedding_dim, vocab_size, bias=True)
        self.emb_layer = nn.Embedding(vocab_size, embedding_dim)
        self.w1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.w2 = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, hidden_states):
        residual = hidden_states 
        logits = self.classify_head(hidden_states)
        log_probs = nn.functional.log_softmax(logits.clone(), dim=-1, dtype=torch.float32).transpose(0, 1)
        setattr(self, "log_probs", log_probs)
        ids = logits.argmax(dim=-1)
        ids_emb = self.emb_layer(ids)

        g = nn.functional.sigmoid(self.w1(residual) + self.w2(ids_emb))
        
        new_state = residual * g + (1.0 - g) * ids_emb
        return new_state

class Wav2Vec2EncoderLayerStableLayerNormOptimal(nn.Module):
    def __init__(self, encoder, gic):
        super(Wav2Vec2EncoderLayerStableLayerNormOptimal, self).__init__()
        self.wrap_encoder = encoder
        self.wrap_gic = gic 

    def forward(self, hidden_states, attention_mask=None, output_attentions=None):
        encoder_out = self.wrap_encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = encoder_out[0]
        hidden_states = self.wrap_gic(hidden_states)
        outputs = (hidden_states, )
        if output_attentions:
            outputs += (attention_mask, )
        return outputs
    

def apply_gic_intermediate(model, K, full_state_dict=None, vocab_size=111, hidden_dim=1024, do_stable=True):
    L = len(model.wav2vec2.encoder.layers)
    inter_idx = [int(i*L/(K+1)) for i in range(1, K+1)]
    new_layers = []
    for i in range(L):
        encoder_layer = lib.Wav2Vec2EncoderLayerStableLayerNorm(model.wav2vec2.encoder.config)
        if do_stable:
            encoder_layer = lib.Wav2Vec2EncoderLayer(model.wav2vec2.encoder.config)
        encoder_layer.load_state_dict(model.wav2vec2.encoder.layers[i].state_dict())
        if i not in inter_idx:
            new_layers.append(encoder_layer)
        else:
            wrap_gic = Wav2Vec2EncoderLayerStableLayerNormOptimal(encoder_layer, GICLayer(vocab_size, hidden_dim))
            new_layers.append(wrap_gic)
    model.wav2vec2.encoder.layers = torch.nn.ModuleList(new_layers)
    if full_state_dict is not None:
        model.load_state_dict(full_state_dict)
    return model