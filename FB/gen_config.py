import json


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


judger_config = Pack(
    data = 'data/negotiate', 
    cuda = True, 
    bsz = 16, 
    clip = 0.5, 
    decay_every = 1, 
    decay_rate = 5.0, 
    dropout = 0.5, 
    init_range = 0.1, 
    lr = 1, 
    max_epoch = 30, 
    min_lr = 0.01, 
    momentum = 0.1, 
    nembed_ctx = 64, 
    nembed_word = 256, 
    nesterov = True, 
    nhid_attn = 256, 
    nhid_ctx = 64, 
    nhid_lang = 128, 
    nhid_sel = 256, 
    nhid_strat = 128, 
    sel_weight = 0.5, 
    model_file = 'sv_model.th', 
    unk_threshold = 20, 
    temperature = 0.1, 
    seed = 1, 
    visual = False, 
    domain = 'object_division', 
    rnn_ctx_encoder = False, 
)

with open('judger_config.json', 'w') as f:
    json.dump(judger_config, f, indent=4)
