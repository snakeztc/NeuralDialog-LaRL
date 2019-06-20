import numpy as np
from latent_dialog import domain
from latent_dialog.metric import MetricsContainer
from latent_dialog.utils import read_lines
from latent_dialog.corpora import EOS, SEL


class Dialog(object):
    """Dialogue runner."""
    def __init__(self, agents, judger, args):
        assert len(agents) == 2
        self.agents = agents
        self.judger = judger
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()

    def _register_metrics(self):
        """Registers valuable metrics."""
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_average('advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        ref_text = ' '.join(read_lines(self.args.ref_text))
        self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 2 and out[0] == SEL and out[1] == EOS

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, ctxs, verbose=True):
        """Runs one instance of the dialogue."""
        assert len(self.agents) == len(ctxs)
        # initialize agents by feeding in the context
        for agent, ctx in zip(self.agents, ctxs):
            agent.feed_context(ctx)

        # choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        begin_name = writer.name
        if verbose:
            print('begin_name = {}'.format(begin_name))

        # initialize BOD utterance for each agent
        writer.bod_init('writer')
        reader.bod_init('reader')

        conv = []
        # reset metrics
        self.metrics.reset()

        first_turn = False
        nturn = 0
        while True:
            nturn += 1
            # produce an utterance
            out, out_words = writer.write() # out: list of word ids, int, len = max_words
            if verbose:
                print('\t{} out_words = {}'.format(writer.name, out_words))

            self.metrics.record('sent_len', len(out))
            self.metrics.record('full_match', out_words)
            self.metrics.record('%s_unique' % writer.name, out_words)

            # append the utterance to the conversation
            conv.append(out_words)
            # make the other agent to read it
            reader.read(out)
            # check if the end of the conversation was generated
            if self._is_selection(out_words) and first_turn:
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                break
            writer, reader = reader, writer
            first_turn = True
            if self.args.max_nego_turn > 0 and nturn >= self.args.max_nego_turn:
                return [], None, [0, 0], None

        choices = []
        # generate choices for each of the agents
        for agent in self.agents:
            agent.transform_dialogue_history()
            # print('\t{} context = {}'.format(agent.name, agent.context))
            # print('\t{} dialogue_text = {}'.format(agent.name, agent.dialogue_text))
            choice = self.judger.choose(agent.context, agent.dialogue_text)
            # print('\t{} choice = {}'.format(agent.name, choice))
            choices.append(choice)

        # print('conv = {}'.format(conv))
        # evaluate the choices, produce agreement and a reward
        agree, rewards = self.domain.score_choices(choices, ctxs)

        if verbose:
            print('ctxs = {}'.format(ctxs))
            print('choices = {}'.format(choices))
            print('agree = {}'.format(agree))
            print('rewards = {}'.format(rewards))

        # perform update, in case if any of the agents is learnable
        for agent, reward in zip(self.agents, rewards):
            agent.update(agree, reward)

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])

        if agree is None:
            agree = False

        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)
        if verbose:
            print('='*50)
            print(self.show_metrics())
            print('='*50)

        stats = dict()
        stats['system_rew'] = self.metrics.metrics['system_rew'].show()
        stats['system_unique'] = self.metrics.metrics['system_unique'].show()
        stats['avg_agree'] = self.metrics.metrics['agree'].show()

        return conv, agree, rewards, stats


class DialogEval(object):
    def __init__(self, agents, judger, args):
        assert len(agents) == 2
        self.agents = agents
        self.judger = judger
        self.args = args
        self.domain = domain.get_domain(args.domain)

    def _is_selection(self, out):
        return len(out) == 2 and out[0] == SEL and out[1] == EOS

    def run(self, ctxs):
        assert len(self.agents) == len(ctxs)
        # initialize agents by feeding in the context
        for agent, ctx in zip(self.agents, ctxs):
            agent.feed_context(ctx)

        # choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        # initialize BOD utterance for each agent
        writer.bod_init('writer')
        reader.bod_init('reader')

        conv = []

        first_turn = False
        nturn = 0
        while True:
            nturn += 1
            # produce an utterance
            out, out_words = writer.write() # out: list of word ids, int, len = max_words
            # print('\t{} out_words = {}'.format(writer.name, out_words))

            # append the utterance to the conversation
            conv.append((writer.name, out_words))
            # make the other agent to read it
            reader.read(out)
            # check if the end of the conversation was generated
            if self._is_selection(out_words) and first_turn:
                break
            writer, reader = reader, writer
            first_turn = True
            if self.args.max_nego_turn > 0 and nturn >= self.args.max_nego_turn:
                return [], None, [0, 0]

        choices = []
        # generate choices for each of the agents
        for agent in self.agents:
            agent.transform_dialogue_history()
            choice = self.judger.choose(agent.context, agent.dialogue_text)
            choices.append(choice)


        # print('ctxs = {}'.format(ctxs))
        # print('choices = {}'.format(choices))
        # evaluate the choices, produce agreement and a reward
        agree, rewards = self.domain.score_choices(choices, ctxs)
        # print('agree = {}'.format(agree))
        # print('rewards = {}'.format(rewards))

        return conv, agree, rewards
