import random


ctxs = []
with open('selfplay.txt', 'r') as f:
    ctx_pair = []
    for line in f:
        ctx = line.strip().split()
        ctx_pair.append(ctx)
        if len(ctx_pair) == 2:
            ctxs.append(ctx_pair)
            ctx_pair = []

random.seed(1)
random.shuffle(ctxs)
ctxs = ctxs[:100] # TODO

with open('selfplay_eval.txt', 'w') as f:
    for pair in ctxs:
        for ctx in pair:
            f.write(' '.join(ctx))
            f.write('\n')
