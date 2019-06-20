import time
import os
import sys

sys.path.append('../')
import json
import torch as th
from latent_dialog.utils import Pack, set_seed
from latent_dialog.corpora import DealCorpus
from latent_dialog.models_deal import HRED
from latent_dialog.main import Reinforce
from latent_dialog.agent_deal import RlAgent, LstmAgent
from latent_dialog.dialog_deal import Dialog, DialogEval
from latent_dialog.domain import ContextGenerator, ContextGeneratorEval

from FB.models.dialog_model import DialogModel as FbDialogModel
from FB.data import WordCorpus as FbWordCorpus
from FB.utils import use_cuda as FB_use_cuda
from latent_dialog.judgment import Judger


def main():
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[START]', start_time, '=' * 30)

    # RL configuration
    env = 'gpu'
    epoch_id = '23'
    folder = '2019-06-20-09-19-39-sl_word'
    simulator_folder = '2019-06-20-09-19-39-sl_word'
    exp_dir = os.path.join('config_log_model', folder, 'rl-' + start_time)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    rl_config = Pack(
        train_path='../data/negotiate/train.txt',
        val_path='../data/negotiate/val.txt',
        test_path='../data/negotiate/test.txt',
        selfplay_path='../data/negotiate/selfplay.txt',
        selfplay_eval_path='../data/negotiate/selfplay_eval.txt',
        sim_config_path=os.path.join('config_log_model', simulator_folder, 'config.json'),
        sim_model_path=os.path.join('config_log_model', simulator_folder, '{}-model'.format(epoch_id)),
        sv_config_path=os.path.join('config_log_model', folder, 'config.json'),
        sv_model_path=os.path.join('config_log_model', folder, '{}-model'.format(epoch_id)),
        rl_config_path=os.path.join(exp_dir, 'rl_config.json'),
        rl_model_path=os.path.join(exp_dir, 'rl_model'),
        ppl_best_model_path=os.path.join(exp_dir, 'ppl_best_model'),
        reward_best_model_path=os.path.join(exp_dir, 'reward_best_model'),
        judger_model_path=os.path.join('../FB', 'sv_model.th'),
        judger_config_path=os.path.join('../FB', 'judger_config.json'),
        record_path=exp_dir,
        record_freq=100,
        use_gpu=env == 'gpu',
        nepoch=4,
        nepisode=0,
        sv_train_freq=4,
        eval_freq=0,
        max_words=100,
        rl_lr=0.1,
        momentum=0.1,
        nesterov=True,
        gamma=0.95,
        rl_clip=1.0,  # TODO it appears this is very very important
        ref_text='../data/negotiate/train.txt',
        domain='object_division',
        max_nego_turn=50,
        random_seed=0,
    )

    # save configuration
    with open(rl_config.rl_config_path, 'w') as f:
        json.dump(rl_config, f, indent=4)

    # set random seed
    set_seed(rl_config.random_seed)

    # load previous supervised learning configuration and corpus
    sv_config = Pack(json.load(open(rl_config.sv_config_path)))
    sim_config = Pack(json.load(open(rl_config.sim_config_path)))

    # TODO revise the use_gpu in the config
    sv_config['use_gpu'] = rl_config.use_gpu
    sim_config['use_gpu'] = rl_config.use_gpu
    corpus = DealCorpus(sv_config)

    # load models for two agents
    # TARGET AGENT
    sys_model = HRED(corpus, sv_config)
    if sv_config.use_gpu:  # TODO gpu -> cpu transfer
        sys_model.cuda()
    sys_model.load_state_dict(th.load(rl_config.sv_model_path, map_location=lambda storage, location: storage))
    # we don't want to use Dropout during RL
    sys_model.eval()
    sys = RlAgent(sys_model, corpus, rl_config, name='System')

    # SIMULATOR we keep usr frozen, i.e. we don't update its parameters
    usr_model = HRED(corpus, sim_config)
    if sim_config.use_gpu:  # TODO gpu -> cpu transfer
        usr_model.cuda()

    usr_model.load_state_dict(th.load(rl_config.sim_model_path, map_location=lambda storage, location: storage))
    usr_model.eval()
    usr_type = LstmAgent
    usr = usr_type(usr_model, corpus, rl_config, name='User')

    # load FB judger model
    judger_config = Pack(json.load(open(rl_config.judger_config_path)))
    judger_config['cuda'] = rl_config.use_gpu
    judger_config['data'] = '../data/negotiate'
    judger_device_id = FB_use_cuda(judger_config.cuda)
    judger_word_corpus = FbWordCorpus(judger_config.data, freq_cutoff=judger_config.unk_threshold, verbose=True)
    judger_model = FbDialogModel(judger_word_corpus.word_dict, judger_word_corpus.item_dict,
                                 judger_word_corpus.context_dict, judger_word_corpus.output_length,
                                 judger_config, judger_device_id)
    if judger_device_id is not None:
        judger_model.cuda(judger_device_id)
    judger_model.load_state_dict(th.load(rl_config.judger_model_path, map_location=lambda storage, location: storage))
    judger_model.eval()
    judger = Judger(judger_model, judger_device_id)

    # initialize communication dialogue between two agents
    dialog = Dialog([sys, usr], judger, rl_config)
    ctx_gen = ContextGenerator(rl_config.selfplay_path)

    # simulation module
    dialog_eval = DialogEval([sys, usr], judger, rl_config)
    ctx_gen_eval = ContextGeneratorEval(rl_config.selfplay_eval_path)

    # start RL
    reinforce = Reinforce(dialog, ctx_gen, corpus, sv_config, sys_model, usr_model, rl_config, dialog_eval,
                          ctx_gen_eval)
    reinforce.run()

    # save sys model
    th.save(sys_model.state_dict(), rl_config.rl_model_path)

    end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[END]', end_time, '=' * 30)


if __name__ == '__main__':
    main()
