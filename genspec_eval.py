# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import json
import pickle

import utils.dataset as dataset
import utils.pretrained_models as prtr
import utils.clicks as clk
import utils.dcg_ips as ips
import utils.ranking as rnk
import utils.evaluate as evl

parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str,
                    help="Model file output from pretrained model.")
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
  loaded_clicks = pickle.load(f)

train_clicks = loaded_clicks['train']
select_clicks = loaded_clicks['select']
validation_clicks = loaded_clicks['validation']

combined_clicks = clk.add_clicks([loaded_clicks['train'],
                              loaded_clicks['select']],
                              safe_copy=True)

click_model = loaded_clicks['click_model']
eta = loaded_clicks['eta']
dataset_name = loaded_clicks['dataset_name']
fold_id = loaded_clicks['dataset_fold']
num_clicks = loaded_clicks['target_num_clicks']

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

pretrain_model = prtr.read_model(args.model_file, data, 1.0)
train_ranking = rnk.data_split_model_rank_and_invert(
                      pretrain_model,
                      data.train
                    )[1]
validation_ranking = rnk.data_split_model_rank_and_invert(
                          pretrain_model,
                          data.validation
                        )[1]

(train_weights,
 train_prop) = clk.compute_weights(
                        train_clicks,
                        data.train,
                        eta)
(_,
 select_prop) = clk.compute_weights(
                        select_clicks,
                        data.train,
                        eta)
(validation_weights,
 validation_prop) = clk.compute_weights(
                        validation_clicks,
                        data.validation,
                        eta)

(combined_weights,
 combined_prop) = clk.compute_weights(
                        combined_clicks,
                        data.train,
                        eta)

epsilon_thres=0.000001
click_order = np.log(np.sum(combined_clicks['num_clicks']))/np.log(10)
if dataset_name == 'Webscope_C14_Set1':
  lr = 10.
  tries = 30+int(10*click_order)
elif dataset_name == 'MSLR-WEB30k':
  lr = 50.
  tries = 30+int(10*click_order)
elif dataset_name == 'istella':
  # Istella
  lr = .01
  tries = 20+int(2*click_order)

print('Training model on training subset for deciding activations.')
train_result = ips.optimize_dcg(
                    args.loss,
                    data,
                    train_weights,
                    validation_weights,
                    learning_rate=lr,
                    learning_rate_decay=1.,
                    trial_epochs=tries,
                    # max_epochs=1000,
                    max_epochs=2,
                    epsilon_thres=epsilon_thres,
                    cutoff=0,
                   )

print('Training model on complete training set for deployment.')
combined_result = ips.optimize_dcg(
                    args.loss,
                    data,
                    combined_weights,
                    validation_weights,
                    learning_rate=lr,
                    learning_rate_decay=1.,
                    trial_epochs=tries,
                    # max_epochs=1000,
                    max_epochs=2,
                    epsilon_thres=epsilon_thres,
                    cutoff=0,
                   )
print('Finished training models.')

general_model = train_result['model']
combined_general_model = combined_result['model']

general_ranking = rnk.data_split_model_rank_and_invert(
                      general_model,
                      data.train
                    )[1]
specialized_ranking = rnk.data_split_rank_and_invert(
                            train_weights,
                            data.train
                          )[1]
combined_general_ranking = rnk.data_split_model_rank_and_invert(
                      combined_general_model,
                      data.train
                    )[1]
combined_specialized_ranking = rnk.data_split_rank_and_invert(
                            combined_weights,
                            data.train
                          )[1]
ideal_ranking = rnk.data_split_rank_and_invert(
                            data.train.label_vector,
                            data.train
                          )[1]

pretrain_docrewards = evl.dcg_reward_per_doc(train_ranking)
general_docrewards = evl.dcg_reward_per_doc(general_ranking)
specialized_docrewards = evl.dcg_reward_per_doc(specialized_ranking)
combined_general_docrewards = evl.dcg_reward_per_doc(combined_general_ranking)
combined_specialized_docrewards = evl.dcg_reward_per_doc(combined_specialized_ranking)

(pretrain_mean,
 pretrain_variance) = evl.mean_variance_reward(
                            pretrain_docrewards,
                            select_prop,
                            select_clicks
                          )

(general_mean,
 general_variance) = evl.mean_variance_reward(
                            general_docrewards,
                            select_prop,
                            select_clicks
                          )

(gendiff_mean,
 gendiff_variance) = evl.mean_variance_reward(
                            general_docrewards-pretrain_docrewards,
                            select_prop,
                            select_clicks
                          )

(pretrain_mean_q,
 pretrain_variance_q) = evl.mean_variance_reward_per_query(
                            pretrain_docrewards,
                            select_prop,
                            select_clicks,
                            data.train,
                          )

(general_mean_q,
 general_variance_q) = evl.mean_variance_reward_per_query(
                            general_docrewards,
                            select_prop,
                            select_clicks,
                            data.train,
                          )

(specialized_mean_q,
 specialized_variance_q) = evl.mean_variance_reward_per_query(
                            specialized_docrewards,
                            select_prop,
                            select_clicks,
                            data.train,
                          )
(specpre_diff_mean_q,
 specpre_diff_variance_q) = evl.mean_variance_reward_per_query(
                            specialized_docrewards - pretrain_docrewards,
                            select_prop,
                            select_clicks,
                            data.train,
                          )
(specgen_diff_mean_q,
 specgen_diff_variance_q) = evl.mean_variance_reward_per_query(
                            specialized_docrewards - general_docrewards,
                            select_prop,
                            select_clicks,
                            data.train,
                          )

true_pretrain_rewards = evl.true_reward_per_query(
                              train_ranking,
                              data.train,
                              binarize=binarize_labels,
                            )
true_general_rewards = evl.true_reward_per_query(
                              general_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )
true_specialized_rewards = evl.true_reward_per_query(
                              specialized_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )
true_combined_general_rewards = evl.true_reward_per_query(
                              combined_general_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )
true_combined_specialized_rewards = evl.true_reward_per_query(
                              combined_specialized_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )
ideal_rewards = evl.true_reward_per_query(
                              ideal_ranking, 
                              data.train,
                              binarize=binarize_labels,
                            )

test_ranking = rnk.data_split_model_rank_and_invert(
                          pretrain_model,
                          data.test
                        )[1]

test_general_ranking = rnk.data_split_model_rank_and_invert(
                      combined_general_model,
                      data.test
                    )[1]
test_ideal_ranking = rnk.data_split_rank_and_invert(
                            data.test.label_vector,
                            data.test
                          )[1]
true_pretrain_test_rewards = evl.true_reward_per_query(
                                  test_ranking,
                                  data.test,
                                  binarize=binarize_labels,
                                )
true_general_test_rewards = evl.true_reward_per_query(
                                  test_general_ranking,
                                  data.test,
                                  binarize=binarize_labels,
                                )
ideal_test_rewards = evl.true_reward_per_query(
                                  test_ideal_ranking,
                                  data.test,
                                  binarize=binarize_labels,
                                )
ideal_test_rewards[np.equal(ideal_test_rewards, 0)] = 1.

def mean_ndcg(rewards, ideal_rewards):
  mask = np.greater(ideal_rewards, 0)
  return np.mean(rewards[mask]/ideal_rewards[mask])

def weighted_mean_ndcg(rewards, ideal_rewards, query_weights):
  mask = np.greater(ideal_rewards, 0)
  return np.sum(query_weights[mask]*(rewards[mask]/ideal_rewards[mask]))

def perc_act(activations, ideal_rewards):
  mask = np.greater(ideal_rewards, 0)
  return np.sum(activations[mask])/float(np.sum(mask))

results = {
    'simulation': {
      'loss function': args.loss,
      'click model': click_model,
      'number of clicks': num_clicks,
      'eta': eta,
      'cutoff': 0,
      'dataset': dataset_name,
      'number of train queries': data.train.num_queries(),
      'number of validation queries': data.validation.num_queries(),
    },
    'common results': {
      'pretrained ndcg mean':  mean_ndcg(true_pretrain_rewards, ideal_rewards),
      'general ndcg mean':     mean_ndcg(true_general_rewards, ideal_rewards),
      'specialized ndcg mean': mean_ndcg(true_specialized_rewards, ideal_rewards),
      'combined general ndcg mean':     mean_ndcg(true_combined_general_rewards, ideal_rewards),
      'combined specialized ndcg mean': mean_ndcg(true_combined_specialized_rewards, ideal_rewards),
      'pretrained unseen ndcg mean':  mean_ndcg(true_pretrain_test_rewards, ideal_test_rewards),
      'general unseen ndcg mean':     mean_ndcg(true_general_test_rewards, ideal_test_rewards),
    },
    'SEA confidence selected': {},
    'GENSPEC confidence selected': {},
  }

print('COMMON STATISTICS')
print('Total number of clicks:', train_clicks['num_clicks'])
print('Mean clicks per query:', train_clicks['num_clicks']/float(data.train.num_queries()))
print('pretrain:', mean_ndcg(true_pretrain_rewards, ideal_rewards))
print('general:', mean_ndcg(true_general_rewards, ideal_rewards))
print('specialized:', mean_ndcg(true_specialized_rewards, ideal_rewards))
print('combined general:', mean_ndcg(true_combined_general_rewards, ideal_rewards))
print('combined specialized:', mean_ndcg(true_combined_specialized_rewards, ideal_rewards))
print('unseen pretrain:', mean_ndcg(true_pretrain_test_rewards, ideal_test_rewards))
print('unseen general:', mean_ndcg(true_general_test_rewards, ideal_test_rewards))

min_prop = 1./np.amax(data.train.query_sizes())
for bound_prob in ([0.0, 0.01, 0.5, 0.75, 0.95]):
  pretrain_cb = evl.confidence_bound(
                             bound_prob,
                             pretrain_variance, 
                             1.,
                             min_prop,
                             select_clicks
                            )
  general_cb = evl.confidence_bound(bound_prob,
                             general_variance, 
                             1.,
                             min_prop,
                             select_clicks
                            )
  gendiff_cb = evl.confidence_bound(bound_prob,
                             gendiff_variance, 
                             1.,
                             min_prop,
                             select_clicks
                            )
  pretrain_cb_q = evl.confidence_bound_per_query(
                        bound_prob,
                        pretrain_variance_q,
                        1.,
                        train_prop,
                        select_clicks,
                        data.train,
                      )

  general_cb_q = evl.confidence_bound_per_query(
                        bound_prob,
                        general_variance_q,
                        1.,
                        train_prop,
                        select_clicks,
                        data.train,
                      )
  specialized_cb_q = evl.confidence_bound_per_query(
                        bound_prob,
                        specialized_variance_q,
                        1.,
                        train_prop,
                        select_clicks,
                        data.train,
                      )

  if general_mean - general_cb > pretrain_mean + general_cb:
    general_model_activated = True
    selected_reward = true_combined_general_rewards
    selected_query_mean = general_mean_q
    selected_query_cb = general_cb_q
    true_selected_rewards = true_combined_general_rewards.copy()
    true_selected_test_rewards = true_general_test_rewards
  else:
    general_model_activated = False
    selected_reward = true_pretrain_rewards
    selected_query_mean = pretrain_mean_q
    selected_query_cb = pretrain_cb_q
    true_selected_rewards = true_pretrain_rewards.copy()
    true_selected_test_rewards = true_pretrain_test_rewards

  selected_upper = selected_query_mean + selected_query_cb
  specialized_lower = specialized_mean_q - specialized_cb_q

  spec_act = np.greater(specialized_lower, selected_upper)
  true_selected_rewards[spec_act] = true_combined_specialized_rewards[spec_act]

  print('CONFIDENCE', bound_prob)
  print('SEA BOUNDS')
  print('confidence bound:', bound_prob)
  print('General model activated:', general_model_activated)
  print('Num queries specialized:', np.sum(spec_act))
  print('Perc. queries specialized:', np.sum(spec_act)/float(spec_act.size))
  print('Cor. Perc. queries specialized:', perc_act(spec_act, ideal_rewards))
  print('selected:', mean_ndcg(true_selected_rewards, ideal_rewards))
  print('unseen selected:', mean_ndcg(true_selected_test_rewards, ideal_test_rewards))

  results['SEA confidence selected'][bound_prob] = {
    'confidence': bound_prob,
    'general activated': general_model_activated,
    'number of activations': int(np.sum(spec_act)),
    'percentage of activations': np.sum(spec_act)/float(data.train.num_queries()),
    'corrected percentage of activations': perc_act(spec_act, ideal_rewards),
    'selected ndcg mean': mean_ndcg(true_selected_rewards, ideal_rewards),
    'selected unseen ndcg mean': mean_ndcg(true_selected_test_rewards, ideal_test_rewards),
  }
  if gendiff_mean - gendiff_cb > 0:
    general_model_activated = True
    selected_reward = true_combined_general_rewards
    selected_query_mean = specgen_diff_mean_q
    selected_query_cb = evl.confidence_bound_per_query(
                              bound_prob,
                              specgen_diff_variance_q,
                              1.,
                              train_prop,
                              select_clicks,
                              data.train,
                            )
    true_selected_rewards = true_combined_general_rewards.copy()
    true_selected_test_rewards = true_general_test_rewards

  else:
    general_model_activated = False
    selected_reward = true_pretrain_rewards
    selected_query_mean = specpre_diff_mean_q
    selected_query_cb = evl.confidence_bound_per_query(
                              bound_prob,
                              specpre_diff_variance_q,
                              1.,
                              train_prop,
                              select_clicks,
                              data.train,
                            )
    true_selected_rewards = true_pretrain_rewards.copy()
    true_selected_test_rewards = true_pretrain_test_rewards

  spec_act = np.greater(selected_query_mean - selected_query_cb, 0)
  true_selected_rewards[spec_act] = true_combined_specialized_rewards[spec_act]

  print('GENSPEC BOUND')
  print('confidence bound:', bound_prob)
  print('General model activated:', general_model_activated)
  print('Num queries specialized:', np.sum(spec_act))
  print('Perc. queries specialized:', np.sum(spec_act)/float(spec_act.size))
  print('Cor. Perc. queries specialized:', perc_act(spec_act, ideal_rewards))
  print('selected:', mean_ndcg(true_selected_rewards, ideal_rewards))
  print('unseen selected:', mean_ndcg(true_selected_test_rewards, ideal_test_rewards))

  results['GENSPEC confidence selected'][bound_prob] = {
    'confidence': bound_prob,
    'general activated': general_model_activated,
    'number of activations': int(np.sum(spec_act)),
    'percentage of activations': np.sum(spec_act)/float(data.train.num_queries()),
    'corrected percentage of activations': perc_act(spec_act, ideal_rewards),
    'selected ndcg mean': mean_ndcg(true_selected_rewards, ideal_rewards),
    'selected unseen ndcg mean': mean_ndcg(true_selected_test_rewards, ideal_test_rewards),
  }

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(results, f)