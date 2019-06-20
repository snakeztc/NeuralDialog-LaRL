import torch as th
import torch.nn as nn
from torch.autograd import Variable
from latent_dialog.base_models import BaseModel
from latent_dialog.corpora import SYS, EOS, PAD
from latent_dialog.utils import INT, FLOAT, LONG, Pack
from latent_dialog.enc2dec.encoders import EncoderRNN, RnnUttEncoder, MlpGoalEncoder
from latent_dialog.nn_lib import IdentityConnector, Bi2UniConnector
from latent_dialog.enc2dec.decoders import DecoderRNN, GEN, GEN_VALID, TEACH_FORCE
from latent_dialog.criterions import NLLEntropy, NLLEntropy4CLF, CombinedNLLEntropy4CLF
import latent_dialog.utils as utils
import latent_dialog.nn_lib as nn_lib
import latent_dialog.criterions as criterions
import numpy as np


class HRED(BaseModel):
    def __init__(self, corpus, config):
        super(HRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.goal_vocab = corpus.goal_vocab
        self.goal_vocab_dict = corpus.goal_vocab_dict
        self.goal_vocab_size = len(self.goal_vocab)
        self.outcome_vocab = corpus.outcome_vocab
        self.outcome_vocab_dict = corpus.outcome_vocab_dict
        self.outcome_vocab_size = len(self.outcome_vocab)
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]

        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size,
                                           k=config.k,
                                           nembed=config.goal_embed_size,
                                           nhid=config.goal_nhid,
                                           init_range=config.init_range)

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=1,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0,
                                      rnn_cell=config.ctx_rnn_cell,
                                      # input_size=self.utt_encoder.output_size+config.goal_nhid, 
                                      input_size=self.utt_encoder.output_size,
                                      hidden_size=config.ctx_cell_size,
                                      num_layers=config.num_layers,
                                      output_dropout_p=config.dropout,
                                      bidirectional=config.bi_ctx_cell,
                                      variable_lengths=False)

        # TODO connector
        if config.bi_ctx_cell:
            self.connector = Bi2UniConnector(rnn_cell=config.ctx_rnn_cell,
                                             num_layer=1,
                                             hidden_size=config.ctx_cell_size,
                                             output_size=config.dec_cell_size)
        else:
            self.connector = IdentityConnector()

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.ctx_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.sys_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        clf = False
        if not clf:
            ctx_lens = data_feed['context_lens']  # (batch_size, )
            ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
            ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)  # (batch_size, max_ctx_len)
            out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
            goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
            batch_size = len(ctx_lens)

            # encode goal info
            goals_h = self.goal_encoder(goals)  # (batch_size, goal_nhid)

            enc_inputs, _, _ = self.utt_encoder(ctx_utts, feats=ctx_confs,
                                                goals=goals_h)  # (batch_size, max_ctx_len, num_directions*utt_cell_size)

            # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
            # enc_last: tuple, (h_n, c_n)
            # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
            # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
            enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

            # get decoder inputs
            dec_inputs = out_utts[:, :-1]
            labels = out_utts[:, 1:].contiguous()

            # pack attention context
            if self.config.dec_use_attn:
                attn_context = enc_outs
            else:
                attn_context = None

            # create decoder initial states
            dec_init_state = self.connector(enc_last)

            # decode
            dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                                   dec_inputs=dec_inputs,
                                                                   # (batch_size, response_size-1)
                                                                   dec_init_state=dec_init_state,  # tuple: (h, c)
                                                                   attn_context=attn_context,
                                                                   # (batch_size, max_ctx_len, ctx_cell_size)
                                                                   mode=mode,
                                                                   gen_type=gen_type,
                                                                   beam_size=self.config.beam_size,
                                                                   goal_hid=goals_h)  # (batch_size, goal_nhid)

            if mode == GEN:
                return ret_dict, labels
            if return_latent:
                return Pack(nll=self.nll(dec_outputs, labels),
                            latent_action=dec_init_state)
            else:
                return Pack(nll=self.nll(dec_outputs, labels))


class GaussHRED(BaseModel):
    def __init__(self, corpus, config):
        super(GaussHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.goal_vocab = corpus.goal_vocab
        self.goal_vocab_dict = corpus.goal_vocab_dict
        self.goal_vocab_size = len(self.goal_vocab)
        self.outcome_vocab = corpus.outcome_vocab
        self.outcome_vocab_dict = corpus.outcome_vocab_dict
        self.outcome_vocab_size = len(self.outcome_vocab)
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.simple_posterior = config.simple_posterior

        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size,
                                           k=config.k,
                                           nembed=config.goal_embed_size,
                                           nhid=config.goal_nhid,
                                           init_range=config.init_range)

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0,
                                      rnn_cell=config.ctx_rnn_cell,
                                      # input_size=self.utt_encoder.output_size+config.goal_nhid,
                                      input_size=self.utt_encoder.output_size,
                                      hidden_size=config.ctx_cell_size,
                                      num_layers=config.num_layers,
                                      output_dropout_p=config.dropout,
                                      bidirectional=config.bi_ctx_cell,
                                      variable_lengths=False)
        # mu and logvar projector
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(config.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size+self.ctx_encoder.output_size, config.y_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.ctx_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.sys_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = criterions.NormKLLoss(unit_average=True)
        self.zero = utils.cast_type(th.zeros(1), FLOAT, self.use_gpu)

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def z2dec(self, last_h, requires_grad):
        p_mu, p_logvar = self.c2z(last_h)
        if requires_grad:
            sample_z = self.gauss_connector(p_mu, p_logvar)
            joint_logpz = None
        else:
            sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
            logprob_sample_z = self.gaussian_logprob(p_mu, p_logvar, sample_z)
            joint_logpz = th.sum(logprob_sample_z.squeeze(0), dim=1)

        dec_init_state = self.z_embedding(sample_z)
        attn_context = None

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        return dec_init_state, attn_context, joint_logpz

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        batch_size = len(ctx_lens)

        # encode goal info
        goals_h = self.goal_encoder(goals)  # (batch_size, goal_nhid)

        enc_inputs, _, _ = self.utt_encoder(ctx_utts, goals=goals_h)
        # (batch_size, max_ctx_len, num_directions*utt_cell_size)

        # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
        # enc_last: tuple, (h_n, c_n)
        # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1),  goals=goals_h)
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1).unsqueeze(0)], dim=2))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z)
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size,
                                                               goal_hid=goals_h)  # (batch_size, goal_nhid)

        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result


class CatHRED(BaseModel):
    def __init__(self, corpus, config):
        super(CatHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.goal_vocab = corpus.goal_vocab
        self.goal_vocab_dict = corpus.goal_vocab_dict
        self.goal_vocab_size = len(self.goal_vocab)
        self.outcome_vocab = corpus.outcome_vocab
        self.outcome_vocab_dict = corpus.outcome_vocab_dict
        self.outcome_vocab_size = len(self.outcome_vocab)
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.simple_posterior = config.simple_posterior

        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size,
                                           k=config.k,
                                           nembed=config.goal_embed_size,
                                           nhid=config.goal_nhid,
                                           init_range=config.init_range)

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0,
                                      rnn_cell=config.ctx_rnn_cell,
                                      # input_size=self.utt_encoder.output_size+config.goal_nhid,
                                      input_size=self.utt_encoder.output_size,
                                      hidden_size=config.ctx_cell_size,
                                      num_layers=config.num_layers,
                                      output_dropout_p=config.dropout,
                                      bidirectional=config.bi_ctx_cell,
                                      variable_lengths=False)
        # mu and logvar projector
        self.c2z = nn_lib.Hidden2Discrete(self.ctx_encoder.output_size, config.y_size, config.k_size,
                                          is_lstm=config.ctx_rnn_cell == 'lstm')

        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Discrete(self.ctx_encoder.output_size + self.utt_encoder.output_size,
                                               config.y_size, config.k_size, is_lstm=False)

        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        self.z_embedding = nn.Linear(config.y_size * config.k_size, config.dec_cell_size, bias=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.ctx_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.sys_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.entropy_loss = criterions.Entropy()

        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss -= self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def z2dec(self, last_h, requires_grad):
        logits, log_qy = self.c2z(last_h)

        if requires_grad:
            sample_y = self.gumbel_connector(logits)
            logprob_z = None
        else:
            idx = th.multinomial(th.exp(log_qy), 1).detach()
            logprob_z = th.sum(log_qy.gather(1, idx))
            sample_y = utils.cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.config.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.config.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            attn_context = None
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))

        return dec_init_state, attn_context, logprob_z

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        batch_size = len(ctx_lens)

        # encode goal info
        goals_h = self.goal_encoder(goals)  # (batch_size, goal_nhid)

        enc_inputs, _, _ = self.utt_encoder(ctx_utts, goals=goals_h)
        # (batch_size, max_ctx_len, num_directions*utt_cell_size)

        # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
        # enc_last: tuple, (h_n, c_n)
        # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1), goals=goals_h)
            logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1).unsqueeze(0)], dim=2))
            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_y = self.gumbel_connector(logits_py)
            else:
                sample_y = self.gumbel_connector(logits_qy)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.config.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.config.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            attn_context = None
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))

        # decode
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size,
                                                               goal_hid=goals_h)  # (batch_size, goal_nhid)


        if mode == GEN:
            return ret_dict, labels
        else:
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            pi_h = self.entropy_loss(log_qy, unit_average=True)
            results = Pack(nll=self.nll(dec_outputs, labels), mi=mi, pi_kl=pi_kl, pi_h=pi_h)

            if return_latent:
                results['latent_action'] = dec_init_state

            return results
