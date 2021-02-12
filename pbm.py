# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import pickle
import time
from multiprocessing import Pool
from scipy.special import gamma

import utils.dataset as dataset
import utils.pretrained_models as prtr
import utils.clicks as clk
import utils.ranking as rnk
import utils.evaluate as evl

parser = argparse.ArgumentParser()
parser.add_argument("click_path", type=str,
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='binarized')
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--eta", type=float,
                    default=1.0,
                    help="Eta parameter for observance probabilities.")
parser.add_argument("--num_proc", type=int,
                    default=1,
                    help="Number of processes to use.")
parser.add_argument("--perc", type=float,
                    default=0.30,
                    help="Percentage of clicks to keep for model selection.")

args = parser.parse_args()

click_model = args.click_model
binarize_labels = 'binarized' in click_model
eta = args.eta
perc = args.perc
num_proc = args.num_proc
assert num_proc >= 0, 'Invalid number of processes: %d' % num_proc

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                  shared_resource = num_proc > 1,
                )

fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

click_list = [10**x for x in range(2, 10)]
click_list += [2*10**x for x in range(2, 9)]
click_list += [5*10**x for x in range(2, 9)]
click_list = sorted(click_list)
click_range = np.array(click_list)

ideal_ranking = rnk.data_split_rank_and_invert(
                            data.train.label_vector,
                            data.train
                          )[1]
ideal_rewards = evl.true_reward_per_query(
                              ideal_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )

print('About to generate clicks')
start = time.time()

def mean_ndcg(rewards, ideal_rewards):
  mask = np.greater(ideal_rewards, 0)
  return np.mean(rewards[mask]/ideal_rewards[mask])

def generate_query_clicks(data_split,
                          prev_docclicks_train,
                          prev_posdocdisplay_train,
                          prev_docclicks_select,
                          prev_posdocdisplay_select,
                          qid,
                          num_query_samples):
  n_docs = data_split.query_size(qid)
  doc_range = np.arange(n_docs)
  labels = data_split.query_labels(qid)
  rel_prob = clk.get_relevance_click_model(click_model)(labels)
  fixed_obs_prob = clk.inverse_rank_prob(doc_range, eta)

  docclicks_train = prev_docclicks_train.copy()
  docclicks_select = prev_docclicks_select.copy()
  posdocdisplay_train = np.reshape(prev_posdocdisplay_train.copy(), (n_docs, n_docs))
  posdocdisplay_select = np.reshape(prev_posdocdisplay_select.copy(), (n_docs, n_docs))

  docclicks_both = docclicks_train + docclicks_select
  posdocdisplay_both = posdocdisplay_train + posdocdisplay_select

  clicks_generated = 0
  train_samples = 0
  select_samples = 0
  for _ in range(num_query_samples):
    obs_estimate = np.sum(fixed_obs_prob[None, :]*posdocdisplay_both, axis=1)
    obs_estimate[np.equal(obs_estimate, 0)] = 1.
    lambdas = docclicks_both/obs_estimate

    ranking = rnk.rank_and_invert(lambdas)[1]
    obs_prob = fixed_obs_prob[ranking]
    c_i = clk.sample_from_click_probs(rel_prob*obs_prob)

    clicks_generated += c_i.size
    docclicks_both[c_i] += 1
    posdocdisplay_both[doc_range, ranking] += 1

    if np.random.uniform() > perc or data_split.name == 'validation':
      train_samples += 1
      docclicks_train[c_i] += 1
      posdocdisplay_train[doc_range, ranking] += 1
    else:
      select_samples += 1
      docclicks_select[c_i] += 1
      posdocdisplay_select[doc_range, ranking] += 1

  return {
      'num_samples': num_query_samples,
      'train_samples': train_samples,
      'select_samples': select_samples,
      'num_clicks': clicks_generated,
      'query_id': qid,
      'train_doc_position_display_freq': posdocdisplay_train,
      'train_clicks_per_doc': docclicks_train,
      'select_doc_position_display_freq': posdocdisplay_select,
      'select_clicks_per_doc': docclicks_select,
      'lambdas': lambdas,
    }

def reduce_results(data_split, results, previous_result=None):
  n_docs = data_split.num_docs()
  query_sizes = data_split.query_sizes()
  if previous_result is None:
    previous_result = {
        'num_clicks': 0,
        'num_queries_sampled': 0,
        'query_freq': np.zeros(data_split.num_queries(), dtype=np.int64),
        'train_query_freq': np.zeros(data_split.num_queries(), dtype=np.int64),
        'select_query_freq': np.zeros(data_split.num_queries(), dtype=np.int64),
        'train_doc_position_display_freq': np.zeros(np.sum(query_sizes**2), dtype=np.int64),
        'train_clicks_per_doc': np.zeros(n_docs, dtype=np.int64),
        'select_doc_position_display_freq': np.zeros(np.sum(query_sizes**2), dtype=np.int64),
        'select_clicks_per_doc': np.zeros(n_docs, dtype=np.int64),
        'train_observance_prop': np.zeros(n_docs, dtype=np.float64),
        'select_observance_prop': np.zeros(n_docs, dtype=np.float64),
        'data_split_name': data_split.name,
        'lambdas': np.zeros(n_docs),
        'bandit_name': 'PBM',
        'click_model': click_model,
        'eta': eta, 
        'dataset_name': data.name,
        'dataset_fold': data.fold_num,
      }

  docpos_ranges = np.zeros(data_split.num_queries()+1, dtype=np.int64)
  docpos_ranges[1:] = np.cumsum(query_sizes**2)
  for res in results:
    qid = res['query_id']
    previous_result['num_clicks'] += res['num_clicks']
    previous_result['num_queries_sampled'] += res['num_samples']
    previous_result['query_freq'][qid] += res['num_samples']
    previous_result['train_query_freq'][qid] += res['train_samples']
    previous_result['select_query_freq'][qid] += res['select_samples']
    s_j, e_j = docpos_ranges[qid], docpos_ranges[qid+1]
    previous_result['train_doc_position_display_freq'][s_j:e_j] = res['train_doc_position_display_freq'].flatten()
    previous_result['select_doc_position_display_freq'][s_j:e_j] = res['select_doc_position_display_freq'].flatten()
    s_i, e_i = data_split.query_range(qid)
    previous_result['train_clicks_per_doc'][s_i:e_i] = res['train_clicks_per_doc']
    previous_result['select_clicks_per_doc'][s_i:e_i] = res['select_clicks_per_doc']
    previous_result['lambdas'][s_i:e_i] = res['lambdas']

    q_n_docs = data_split.query_size(qid)
    fixed_obs_prob = clk.inverse_rank_prob(np.arange(q_n_docs), eta)
    if res['train_samples'] > 0:
      previous_result['train_observance_prop'][s_i:e_i] = (
            np.sum(res['train_doc_position_display_freq']
                   /float(previous_result['train_query_freq'][qid])
                   *fixed_obs_prob[None, :],
                   axis=1))
    if res['select_samples'] > 0:
      previous_result['select_observance_prop'][s_i:e_i] = (
            np.sum(res['select_doc_position_display_freq']
                   /float(previous_result['select_query_freq'][qid])
                   *fixed_obs_prob[None, :],
                   axis=1))
  return previous_result

def multi_optimize(m_args):
  qid, num_samples, clicks, docpos, clicks2, docpos2, split_name = m_args
  if split_name == 'train':
    data_split = data.train
  elif split_name == 'validation':
    data_split = data.validation
  elif split_name == 'test':
    data_split = data.test
  return generate_query_clicks(
                          data_split,
                          clicks,
                          docpos,
                          clicks2,
                          docpos2,
                          qid,
                          num_samples)

def display_ndcg(data_split, lambdas, qid, k=None):
  n_docs = data_split.query_size(qid)
  s_i, e_i = data_split.query_range(qid)
  ranking = rnk.rank_and_invert(lambdas[s_i:e_i])[1]
  return {
      'query_id': qid,
      'ranking': ranking,
    }

def reduce_rankings(data_split, rankings):
  n_docs = data_split.num_docs()
  query_sizes = data_split.query_sizes()
  inverted_ranking = np.zeros(n_docs, dtype=np.int64)
  for res in rankings:
    qid = res['query_id']
    s_i, e_i = data_split.query_range(qid)
    inverted_ranking[s_i:e_i] = res['ranking']
  return inverted_ranking

def multi_display_rank(m_args):
  qid, preranking, split_name = m_args
  if split_name == 'train':
    data_split = data.train
  elif split_name == 'validation':
    data_split = data.validation
  elif split_name == 'test':
    data_split = data.test
  return display_ndcg(data_split,
                      preranking,
                      qid)

if num_proc > 1:
  pool = Pool(processes=args.num_proc)

def sample_clicks(data_split, sample_size, cur_result=None):
  click_query_ratio = 1./1.6

  n_docs = data_split.num_docs()
  query_sizes = data_split.query_sizes()
  docpos_ranges = np.zeros(data_split.num_queries()+1, dtype=np.int64)
  docpos_ranges[1:] = np.cumsum(query_sizes**2)

  while cur_result is None or cur_result['num_clicks'] < sample_size:
    if cur_result is None:
      next_sample = int(sample_size*click_query_ratio)
      squared_sum = np.sum(query_sizes**2)
      docclicks_train = np.zeros(n_docs, dtype=np.int64)
      docposdisplay_train = np.zeros(squared_sum, dtype=np.int64)
      docclicks_select = np.zeros(n_docs, dtype=np.int64)
      docposdisplay_select = np.zeros(squared_sum, dtype=np.int64)
    else:
      if cur_result['num_clicks'] > data_split.num_queries():
        click_query_ratio = float(cur_result['num_queries_sampled'])/cur_result['num_clicks']
      next_sample = int(0.9*(sample_size - cur_result['num_clicks'])*click_query_ratio)
      docclicks_train = cur_result['train_clicks_per_doc']
      docposdisplay_train = cur_result['train_doc_position_display_freq']
      docclicks_select = cur_result['select_clicks_per_doc']
      docposdisplay_select = cur_result['select_doc_position_display_freq']

    next_sample = max(next_sample, 1)

    query_dist = np.zeros(data_split.num_queries(), dtype=np.int64)
    query_sample = np.random.choice(data_split.num_queries(), size=next_sample)
    bincount = np.bincount(query_sample)
    query_dist[:bincount.shape[0]] = bincount

    def start(qid):
      return data_split.query_range(qid)[0]
    def end(qid):
      return data_split.query_range(qid)[1]

    jobs = [(qid, num_samples,
             docclicks_train[start(qid):end(qid)],
             docposdisplay_train[docpos_ranges[qid]:docpos_ranges[qid+1]],
             docclicks_select[start(qid):end(qid)],
             docposdisplay_select[docpos_ranges[qid]:docpos_ranges[qid+1]],
             data_split.name)
            for qid, num_samples in zip(np.arange(data_split.num_queries()),
                                        query_dist) if num_samples > 0]

    if num_proc < 2:
      results = []
      for qid, num_samples, _, _, _, _, _ in jobs:
        results.append(
          generate_query_clicks(data_split,
                                docclicks_train[start(qid):end(qid)],
                                docposdisplay_train[docpos_ranges[qid]:docpos_ranges[qid+1]],
                                docclicks_select[start(qid):end(qid)],
                                docposdisplay_select[docpos_ranges[qid]:docpos_ranges[qid+1]],
                                qid,
                                num_samples,
                                ))
    else:
      results = pool.map(multi_optimize, jobs)

    cur_result = reduce_results(data_split, results, previous_result=cur_result)

  return cur_result

def sample_display_ndcg(data_split, lambdas):
  jobs = [(qid, lambdas, data_split.name)
            for qid in np.arange(data_split.num_queries())]

  if num_proc < 2:
    results = []
    for qid, lambdas, _ in jobs:
      results.append(
                 display_ndcg(data_split,
                              lambdas,
                              qid))
  else:
    results = pool.map(multi_display_rank, jobs)

  display_ranking = reduce_rankings(data_split, results)
  display_rewards = evl.true_reward_per_query(
                          display_ranking,
                          data_split,
                          binarize=binarize_labels)
  return mean_ndcg(display_rewards, ideal_rewards)

train_result = None
vali_result = None
for n in click_range:
  train_result = sample_clicks(data.train, n, train_result)
  train_result['display_ndcg'] = sample_display_ndcg(data.train, train_result['lambdas'])
  vali_result = sample_clicks(data.validation, int(n*0.15), vali_result)
  result = {
        'train': train_result.copy(),
        'validation': vali_result.copy(),
        'target_num_clicks': n
      }
  del result['train']['train_doc_position_display_freq']
  del result['train']['select_doc_position_display_freq']
  del result['validation']['train_doc_position_display_freq']
  del result['validation']['select_doc_position_display_freq']

  file_path = args.click_path + '%sclicks.pkl' % n
  print('Saving result to:', file_path)
  with open(file_path, 'wb') as f:
    pickle.dump(result, f, protocol=4)
  print('Clicks per second:', (train_result['num_clicks']+vali_result['num_clicks'])/float(time.time() - start))

print('Time past for generating train clicks: %d seconds' % (time.time() - start))
