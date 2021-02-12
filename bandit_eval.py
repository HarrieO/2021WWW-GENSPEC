# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import pickle
import time
import json

import utils.dataset as dataset
import utils.pretrained_models as prtr
import utils.clicks as clk
import utils.dcg_ips as ips
import utils.ranking as rnk
import utils.evaluate as evl

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str,
                    help="Path to click input.")
parser.add_argument("output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--loss", type=str,
                    help="Loss to optimize.",
                    default='dcg2')
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")

args = parser.parse_args()

print('Reading clicks from: %s' % args.input_file)
with open(args.input_file, 'rb') as f:
  innp = pickle.load(f)

train_clicks = innp['train']
validation_clicks = innp['validation']

train_doc_clicks = {'clicks_per_doc': train_clicks['train_clicks_per_doc']}
select_doc_clicks = {'clicks_per_doc': train_clicks['select_clicks_per_doc']}
validation_doc_clicks = {'clicks_per_doc': validation_clicks['train_clicks_per_doc']}

bandit_lambdas = train_clicks['lambdas']

click_model = train_clicks['click_model']
eta = train_clicks['eta']
dataset_name = train_clicks['dataset_name']
fold_id = train_clicks['dataset_fold']

binarize_labels = 'binarized' in click_model
num_proc = 0

data = dataset.get_dataset_from_json_info(
                  dataset_name,
                  args.dataset_info_path,
                  shared_resource = False,
                )
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

def process_loaded_clicks(loaded_clicks, data_split):
  train_weights = compute_weights(
                      data_split,
                      loaded_clicks['train_query_freq'],
                      loaded_clicks['train_clicks_per_doc'],
                      loaded_clicks['train_observance_prop'],
                    )
  select_weights = compute_weights(
                      data_split,
                      loaded_clicks['select_query_freq'],
                      loaded_clicks['select_clicks_per_doc'],
                      loaded_clicks['select_observance_prop'],
                    )
  train_mask = np.equal(loaded_clicks['train_observance_prop'], 0).astype(np.float64)
  select_mask = np.equal(loaded_clicks['select_observance_prop'], 0).astype(np.float64)
  return (train_weights,
          1./(loaded_clicks['train_observance_prop'] + train_mask),
          select_weights,
          1./(loaded_clicks['select_observance_prop'] + select_mask))

def compute_weights(data_split, query_freq, clicks_per_doc, observe_prop):
  doc_weights = np.zeros(data_split.num_docs())
  for qid in np.arange(data_split.num_queries()):
    q_freq = query_freq[qid]
    if q_freq <= 0:
      continue
    s_i, e_i = data_split.query_range(qid)
    q_click = clicks_per_doc[s_i:e_i]
    q_obs_prop = observe_prop[s_i:e_i]
    if np.sum(q_click) <= 0:
      continue

    click_prob = q_click.astype(np.float64)/np.amax(q_click)
    unnorm_weights = click_prob/q_obs_prop
    if np.sum(unnorm_weights) == 0:
      norm_weights = unnorm_weights
    else:
      norm_weights = unnorm_weights/np.sum(unnorm_weights)
    norm_weights *= float(q_freq)/np.sum(query_freq)
    doc_weights[s_i:e_i] = norm_weights
  return doc_weights

(train_weights, train_prop,
 select_weights, select_prop) = process_loaded_clicks(train_clicks, data.train)
validation_weights, validation_prop, _, _ = process_loaded_clicks(validation_clicks, data.validation)

ideal_ranking = rnk.data_split_rank_and_invert(
                            data.train.label_vector,
                            data.train
                          )[1]
bandit_ranking = rnk.data_split_rank_and_invert(
                            bandit_lambdas,
                            data.train
                          )[1]

ideal_rewards = evl.true_reward_per_query(
                              ideal_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )
true_bandit_rewards = evl.true_reward_per_query(
                              bandit_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )

def mean_ndcg(rewards, ideal_rewards):
  mask = np.greater(ideal_rewards, 0)
  return np.mean(rewards[mask]/ideal_rewards[mask])

results = {
    'simulation': {
      'loss function': args.loss,
      'click model': click_model,
      'real number of clicks': int(train_clicks['num_clicks']),
      'eta': float(eta),
      'cutoff': 0,
      'dataset': dataset_name,
      'number of train queries': data.train.num_queries(),
      'number of validation queries': data.validation.num_queries(),
    },
    'common results': {
      'bandit ndcg mean': mean_ndcg(true_bandit_rewards, ideal_rewards),
      'bandit display ndcg mean': train_clicks['display_ndcg'],
    },
  }


if 'target_num_clicks' in train_clicks:
  results['simulation']['number of clicks'] = int(train_clicks['target_num_clicks'])
else:
  results['simulation']['number of clicks'] = int(train_clicks['num_clicks'])

if train_clicks['bandit_name'] == 'hotfix':
  results['simulation']['bandit_name'] = 'hotfix'
  if train_clicks['k']:
    results['simulation']['bandit_display_name'] = 'hotfix_at_%s' % train_clicks['k']
    results['simulation']['k'] = int(train_clicks['k'])
  else:
    results['simulation']['bandit_display_name'] = 'hotfix_all'
    results['simulation']['k'] = None
else:
  results['simulation']['bandit_display_name'] = train_clicks['bandit_name']
  results['simulation']['bandit_name'] = train_clicks['bandit_name']


print('COMMON STATISTICS')
print('Total number of clicks:', train_clicks['num_clicks'])
print('Mean clicks per query:', train_clicks['num_clicks']/float(data.train.num_queries()))
print('Bandit NDCG:', mean_ndcg(true_bandit_rewards, ideal_rewards))
print('Bandit Displayed NDCG:', train_clicks['display_ndcg'])

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(results, f)
