import cPickle
import os
import logging
import sys
import time
logger = logging.getLogger(__name__)

import numpy as np
import hyperopt
from hyperopt.base import trials_from_docs
from hyperopt.mongoexp import MongoTrials
from hyperopt.utils import json_call

class InterleaveAlgo(hyperopt.BanditAlgo):
    """Interleaves calls to `num_procs` independent Trials sets

    This class is implemented carefully to work around some awkwardness
    in the design of hyperopt.Trials.  The purpose of this banditalgo is to
    facilitate running several independent BanditAlgos at once.  You might
    want to do this if you are trying to compare random search to an optimized
    search for example.

    Trial documents inserted by this suggest function can be tagged with
    identifying information, but trial documents inserted in turn by the
    Bandit.evaluate() functions themselves will not be tagged, except by their
    experiment key (exp_key). So this class uses the exp_key to keep each
    experiment distinct.  Every sub_algo passed to the constructor requires a
    corresponding exp_key. If you want to get tricky, you can combine
    sub-experiments by giving them identical keys. As a consequence of this
    strategy, this class REQUIRES AN UNRESTRICTED VIEW of the trials object
    used to make suggestions, so the trials object CANNOT have an exp_key
    of its own.

    The InterleaveAlgo works on the basis of growing the sub-experiments at
    the same pace. On every call to suggest(), this function counts up the
    number of non-error jobs in each sub-experiment, and asks the sub_algo
    corresponding to the smallest sub-experiment to propose a new document.

    The InterleaveAlgo stops when all of the sub_algo.suggest methods have
    returned `hyperopt.StopExperiment`.

    """
    def __init__(self, sub_algos, sub_exp_keys, priorities=None, **kwargs):
        if priorities is None:
            priorities = np.ones(len(sub_algos))
        else:
            priorities = np.array(priorities)
        assert (priorities >= 0).all()
        priorities = priorities.astype('float32') / priorities.sum()
        self.priorities = priorities

        hyperopt.BanditAlgo.__init__(self, sub_algos[0].bandit, **kwargs)
        # XXX: assert all bandits are the same
        self.sub_algos = sub_algos
        self.sub_exp_keys = sub_exp_keys
        if len(sub_algos) != len(sub_exp_keys):
            raise ValueError('algos and keys should have same len')
        # -- will be rebuilt naturally if experiment is continued
        self.stopped = set()

    def suggest(self, new_ids, trials):
        assert trials._exp_key is None # -- see explanation above
        sub_trials = [trials.view(exp_key, refresh=False)
                for exp_key in self.sub_exp_keys]
        # -- views are not refreshed
        states_that_count = [
                hyperopt.JOB_STATE_NEW,
                hyperopt.JOB_STATE_RUNNING,
                hyperopt.JOB_STATE_DONE]
        counts = [st.count_by_state_unsynced(states_that_count)
                for st in sub_trials]
        logger.info('counts: %s' % str(counts))
        new_docs = []
        priorities = self.priorities
        for new_id in new_ids:
            weighted_counts = np.array(counts) / priorities
            pref = np.argsort(weighted_counts)
            # -- try to get one of the sub_algos to make a suggestion
            #    for new_id, in order of sub-experiment size
            for active in pref:
                if active not in self.stopped:
                    sub_algo = self.sub_algos[active]
                    sub_trial = sub_trials[active]
                    # XXX This may well transfer data... make sure that's OK
                    #     In future consider adding a refresh=False
                    #     to constructor, to prevent this transfer.
                    sub_trial.refresh()
                    t_before = time.time()
                    smth = sub_algo.suggest([new_id], sub_trial)
                    t_after = time.time()
                    logger.info('%s.suggest() took %f seconds' % (
                            sub_algo, t_after - t_before))
                    if smth is hyperopt.StopExperiment:
                        logger.info('stopping experiment (%i: %s)' %
                                    (active, sub_algo))
                        self.stopped.add(active)
                    elif smth:
                        logger.info('suggestion %i from (%i: %s)' %
                                    (new_id, active, sub_algo))
                        new_doc, = smth
                        counts[active] += 1
                        new_docs.append(new_doc)
                        break
                    else:
                        if list(smth) != []:
                            raise ValueError('bad suggestion',
                                    (sub_algo, smth))
        if len(self.stopped) == len(self.sub_algos):
            return hyperopt.StopExperiment
        else:
            return new_docs


def suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names,
                     exp_keys, N, bandit_args_list, bandit_kwargs_list,
                     bandit_algo_args_list, bandit_algo_kwargs_list,
                     mql=1, refresh=False):
    port = int(port)
    trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname),
                        refresh=False)
    algos = []
    for bn, ban, ba, bk, baa, bak, ek in zip(bandit_names, bandit_algo_names,
            bandit_args_list, bandit_kwargs_list, bandit_algo_args_list, bandit_algo_kwargs_list,
                                             exp_keys):
        bandit = json_call(bn, ba, bk)
        subtrials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname),
                         refresh=False, exp_key=ek)
        if ba or bk:
            subtrials.attachments['bandit_data_' + ek] = cPickle.dumps((bn, ba, bk))
            bak['cmd'] = ('driver_attachment', 'bandit_data_' + ek)
        else:
            bak['cmd'] = ('bandit_json evaluate', bn)
        algo = json_call(ban, (bandit,) + baa, bak)
        algos.append(algo)

    algo = InterleaveAlgo(algos, exp_keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.max_queue_len = mql
    if N is not None:
        exp.run(N, block_until_done=True)
    else:
        return exp
