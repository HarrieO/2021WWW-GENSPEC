# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import time
import utils.ranking as rnk

def calc_true_dcg_loss(ranking_model, data_split, cutoff=0):
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)

  result = 0.
  denom = 0.
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]
    q_labels = data_split.query_labels(qid)

    label_filter = np.greater(q_labels, 2)
    inv_ranking = rnk.rank_and_invert(q_scores)[1]
    dcg_gain = 1./np.log2(inv_ranking[label_filter]+2.)
    if cutoff > 0:
      dcg_gain[np.greater_equal(inv_ranking[label_filter], cutoff)] = 0.

    result += np.sum(dcg_gain)
    denom += np.sum(label_filter)

  return -result/denom

def estimate_dcg_loss(ranking_model,
                      data_split,
                      click_weights,
                      cutoff=0):
  inv_prop = click_weights
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)

  result = 0.
  denom = 0.
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]
  
    q_inv_prop = inv_prop[s_i:e_i]
    inv_ranking = rnk.rank_and_invert(q_scores)[1]
    dcg_gain = 1./np.log2(inv_ranking+2.)
    if cutoff > 0:
      dcg_gain[np.greater_equal(inv_ranking, cutoff)] = 0.
    result += np.sum(dcg_gain*q_inv_prop)

  return -result

def linear_diff_grad(scores):
  return np.less_equal(scores[:, None], scores[None, :]+1.).astype(float)

def sigmoid_diff_grad(scores):
  score_diff = scores[:, None] - scores[None, :]
  # for stability we cap this at 700
  score_diff = np.maximum(score_diff, -700.)
  exp_diff = np.exp(-score_diff)
  return 1./((1+exp_diff)*np.log(2.))*exp_diff

def optimize_dcg(
             loss_name,
             data,
             train_weights,
             validation_weights,
             learning_rate,
             trial_epochs,
             max_epochs=50,
             epsilon_thres=0.0001,
             learning_rate_decay=0.97,
             cutoff=5):

  est_loss_fn = estimate_dcg_loss
  true_loss_fn = calc_true_dcg_loss

  inv_prop = train_weights

  best_model = np.zeros(data.train.datafold.num_features)
  best_loss = np.inf
  best_epoch = 0
  pivot_loss = np.inf
  model = np.zeros(data.train.datafold.num_features)

  start_time = time.time()

  num_docs = data.train.num_docs()
  doc_feat = data.train.feature_matrix

  epoch_i = 0
  cur_loss = est_loss_fn(model,
                         data.validation,
                         validation_weights,
                         cutoff)

  true_loss = true_loss_fn(model,
                           data.validation,
                           cutoff)
  print('Epoch: %d Estimated Loss: %0.05f True Validation Loss: %0.05f' % (epoch_i, cur_loss, true_loss))


  stop_epoch = trial_epochs
  while epoch_i < min(stop_epoch, max_epochs):
    permutation = np.random.permutation(data.train.num_queries())
    for qid in permutation:
      q_docs = data.train.query_feat(qid)
      q_scores = np.dot(q_docs, model)
      n_docs = q_docs.shape[0]

      s_i, e_i = data.train.doclist_ranges[qid:qid+2]
      q_prop = inv_prop[s_i:e_i]

      if loss_name in ['monotonic',
                       'log_monotonic',
                       'relevant_rank',
                       'log_relevant_rank']:

        if loss_name in ['monotonic', 'relevant_rank']:
          diff_grad = linear_diff_grad(q_scores)
        elif loss_name in ['log_monotonic', 'log_relevant_rank']:
          diff_grad = sigmoid_diff_grad(q_scores)
        activation_gradient = -diff_grad*q_prop[:, None]
        
        if 'monotonic' in loss_name:
          if loss_name == 'monotonic':
            up_rank = np.sum(np.maximum(1 - (q_scores[:, None] - q_scores[None, :]), 0), axis=1)
          elif loss_name == 'log_monotonic':
            score_diff = q_scores[:, None] - q_scores[None, :]
            up_rank = np.sum(
                np.log2(1 + np.exp(-score_diff)),
              axis=1)

          dcg_weights = 1./(np.log2(up_rank+1.)**2*np.log(2)*(up_rank+1))
          activation_gradient *= dcg_weights[:, None]

      elif loss_name in ['dcg2atk', 'dcg2', 'trcdcg2']:
        q_inv = rnk.rank_and_invert(q_scores)[1]

        prop_diff = q_prop[:, None] - q_prop[None, :]
        prop_mask = np.less_equal(prop_diff, 0.)

        if loss_name == 'trcdcg2':
          rnk_vec = np.less(q_inv, cutoff)
          rnk_mask = np.logical_or(rnk_vec[:, None],
                                    rnk_vec[None, :])

          prop_mask = np.logical_or(np.logical_not(rnk_mask), prop_mask)

        rank_diff = np.abs(q_inv[:, None] - q_inv[None, :])
        rank_diff[prop_mask] = 1.

        disc_upp = 1. / np.log2(rank_diff+1.)
        disc_low = 1. / np.log2(rank_diff+2.)
        if loss_name == 'dcg2atk':
          disc_upp[np.greater(rank_diff, cutoff)] = 0.
          disc_low[np.greater(rank_diff, cutoff-1)] = 0.

        pair_w = disc_upp - disc_low
        pair_w *= np.abs(prop_diff)
        pair_w[prop_mask] = 0.
        
        score_diff = q_scores[:, None] - q_scores[None, :]
        score_diff[prop_mask] = 0.
        safe_diff = np.minimum(-score_diff, 500)
        act = 1./(1 + np.exp(safe_diff))
        act[prop_mask] = 0.
        safe_exp = pair_w - 1.
        safe_exp[prop_mask] = 0.

        log2_grad = 1./(act**pair_w*np.log(2))
        power_grad = pair_w*(act)**safe_exp
        sig_grad = act*(1-act)

        activation_gradient = -log2_grad*power_grad*sig_grad

      np.fill_diagonal(activation_gradient,
                         np.diag(activation_gradient)
                         - np.sum(activation_gradient, axis=1))

      doc_weights = np.sum(activation_gradient, axis=0)

      gradient = np.sum(q_docs * doc_weights[:, None], axis=0)
      
      model += (learning_rate*gradient
                *(learning_rate_decay**epoch_i))

    epoch_i += 1
    cur_loss = est_loss_fn(model,
                           data.validation,
                           validation_weights,
                           cutoff)

    true_loss = true_loss_fn(model,
                             data.validation,
                             cutoff)
    print('Epoch: %d Estimated Loss: %0.05f True Validation Loss: %0.05f' % (epoch_i, cur_loss, true_loss))

    if cur_loss < best_loss:
      best_model = model
      best_loss = cur_loss
      best_epoch = epoch_i
      if pivot_loss - cur_loss > epsilon_thres:
        pivot_loss = cur_loss
        stop_epoch = epoch_i + trial_epochs

  true_loss = true_loss_fn(best_model,
                           data.validation,
                           cutoff)
  
  result = {
      'model': best_model,
      'estimated_loss': best_loss,
      'true_loss': true_loss,
      'epoch': best_epoch,
      'total_time_spend': time.time()-start_time,
      'time_per_epoch': (time.time()-start_time)/float(epoch_i),
      'learning_rate': learning_rate,
      'trial_epochs': trial_epochs,
      'learning_rate_decay': learning_rate_decay,
    }
  return result