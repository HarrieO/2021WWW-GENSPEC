# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np

def dcg_reward_per_doc(inverted_ranking):
  return 1./np.log2(inverted_ranking+2.)

def true_reward_per_query(inverted_ranking, data_split, binarize=False):
  discounts = dcg_reward_per_doc(inverted_ranking)
  if binarize:
    numerator = np.greater(data_split.label_vector, 2).astype(np.float64)
  else:
    numerator = 2.**data_split.label_vector - 1.
  doc_rewards = numerator*discounts
  n_queries = data_split.num_queries()
  result = np.zeros(n_queries, dtype=np.float64)
  for qid in np.arange(n_queries):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    result[qid] = np.sum(doc_rewards[s_i:e_i])
  return result

def mean_reward_per_query(doc_rewards, doc_weights, clicks, data_split):
  doc_clicks = clicks['clicks_per_doc']
  n_clicks = float(np.sum(doc_clicks))
  n_queries = data_split.num_queries()
  mean_values = np.zeros(n_queries, dtype=np.float64)
  for qid in np.arange(n_queries):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    mean_values[qid] = np.sum(doc_rewards[s_i:e_i]
                              *doc_weights[s_i:e_i])
  return mean_values

def mean_reward(doc_rewards, doc_weights, clicks):
  doc_clicks = clicks['clicks_per_doc']
  n_clicks = float(np.sum(doc_clicks))
  return np.sum(doc_rewards*doc_weights)

def mean_variance_reward(doc_rewards, doc_prop, clicks):
  doc_clicks = clicks['clicks_per_doc']
  n_clicks = float(np.sum(doc_clicks))
  click_weights = doc_clicks/n_clicks
  mean = np.sum(click_weights*(doc_rewards/doc_prop))
  variance = np.sum(click_weights*(doc_rewards/doc_prop - mean)**2)
  return mean, variance

def mean_variance_reward_click_pattern(doc_rewards, doc_prop, click_patterns):
  n_patterns = len(click_patterns)
  pattern_rewards = np.zeros(n_patterns, dtype=np.float64)
  pattern_freq = np.zeros(n_patterns, dtype=np.float64)

  for i, (k,v) in enumerate(click_patterns.items()):
    if k != ():
      k = np.array(k, dtype=np.int64)
      pattern_rewards[i] = np.sum(doc_rewards[k]/doc_prop[k], dtype=np.float64)
      pattern_freq[i] = v
  
  n_samples = np.sum(pattern_freq, dtype=np.int64)
  freq_w = pattern_freq/n_samples.astype(np.float64)
  mean = np.sum(pattern_rewards*freq_w, dtype=np.float64)
  variance = np.sum(freq_w*(pattern_rewards-mean)**2, dtype=np.float64)
  return mean, variance, n_samples

def confidence_bound(conf_prob, variance, max_reward, min_prop, clicks):
  doc_clicks = clicks['clicks_per_doc']
  if conf_prob <= 0.:
    return 0.
  n_clicks = float(np.sum(doc_clicks))
  b = max_reward/float(min_prop)
  if n_clicks > 1:
    bf_sqrt = np.log(2./(1.-conf_prob))*variance*(n_clicks/(n_clicks-1.))*n_clicks*2.
    right_part = np.sqrt(bf_sqrt)/n_clicks
    left_part = 7*b*np.log(2./(1.-conf_prob))/(3*(n_clicks-1))
    return left_part + right_part
  else:
    return np.inf

def max_reward_click_pattern(doc_rewards, doc_prop, data_split):
  n_queries = data_split.num_queries()
  max_rewards = np.zeros(n_queries, dtype=np.float64)
  for qid in range(n_queries):
    s_i, e_i = data_split.query_range(qid)
    q_rewards = doc_rewards[s_i:e_i]/doc_prop[s_i:e_i]
    pos_reward = np.sum(q_rewards[np.greater(q_rewards, 0)])
    neg_reward = np.sum(np.abs(q_rewards[np.less(q_rewards, 0)]))
    max_rewards[qid] = max(pos_reward, neg_reward)
  return max_rewards

def confidence_bound_click_pattern(conf_prob, variance, max_reward, n_samples):
  if conf_prob <= 0.:
    return 0.
  b = max_reward
  if n_samples > 1:
    bf_sqrt = np.log(2./(1.-conf_prob))*variance*(n_samples/(n_samples-1.))*n_samples*2.
    right_part = np.sqrt(bf_sqrt)/n_samples
    left_part = 7*b*np.log(2./(1.-conf_prob))/(3*(n_samples-1))
    return left_part + right_part
  else:
    return np.inf

def mean_variance_reward_per_query(doc_rewards, doc_prop, clicks, data_split):
  doc_clicks = clicks['clicks_per_doc']
  n_queries = data_split.num_queries()

  means = np.zeros(n_queries, dtype=np.float64)
  variances = np.zeros(n_queries, dtype=np.float64)
  for qid in np.arange(n_queries):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    n_clicks = float(np.sum(doc_clicks[s_i:e_i]))
    if n_clicks > 1:
      click_weights = doc_clicks[s_i:e_i]/n_clicks
      means[qid] = np.sum(click_weights*(doc_rewards[s_i:e_i]/doc_prop[s_i:e_i]))
      squared_diff = (doc_rewards[s_i:e_i]/doc_prop[s_i:e_i]-means[qid])**2
      variances[qid] = np.sum(click_weights*squared_diff)
    else:
      variances[qid] = np.inf
  return means, variances

def mean_variance_reward_click_pattern_per_query(doc_rewards,
                                                 doc_prop,
                                                 click_patterns,
                                                 query_freq,
                                                 data_split):
  n_queries = data_split.num_queries()

  n_patterns = len(click_patterns)
  pattern_rewards = np.zeros(n_patterns, dtype=np.float64)
  pattern_freq = np.zeros(n_patterns, dtype=np.float64)
  pattern_qid = np.zeros(n_patterns, dtype=np.int64)

  for i, (k,v) in enumerate(click_patterns.items()):
    if k != ():
      k = np.array(k, dtype=np.int64)
      pattern_rewards[i] = np.sum(doc_rewards[k]/doc_prop[k], dtype=np.float64)
      pattern_qid[i] = n_queries - np.sum(
                          np.greater(data_split.doclist_ranges, k[0])
                        )
      pattern_freq[i] = v
    else:
      pattern_qid[i] = -1

  means = np.zeros(n_queries, dtype=np.float64)
  variances = np.full(n_queries, np.inf, dtype=np.float64)
  query_samples = np.zeros(n_queries, dtype=np.int64)
  for qid in np.unique(pattern_qid):
    if qid == -1:
      continue
    s_i, e_i = data_split.query_range(qid)
    q_mask = np.equal(pattern_qid, qid)
    q_rewards = pattern_rewards[q_mask]
    q_freq = pattern_freq[q_mask]
    total_click_freq = np.sum(q_freq)
    no_click_freq = query_freq[qid] - np.sum(q_freq)

    pattern_weights = q_freq.astype(np.float64)/total_click_freq
    # pattern_weights = q_freq.astype(np.float64)/query_freq[qid]
    q_mean = np.sum(q_rewards*pattern_weights, dtype=np.float64)
    q_variance = np.sum(pattern_weights*(q_mean - q_rewards)**2., dtype=np.float64)
    # q_variance += q_mean**2.*(no_click_freq/float(query_freq[qid]))

    means[qid] = q_mean
    variances[qid] = q_variance
    query_samples[qid] = total_click_freq
  return means, variances, query_samples

def confidence_bound_per_query(conf_prob, variance,
                               max_reward, doc_prop,
                               clicks, data_split):
  doc_clicks = clicks['clicks_per_doc']
  if conf_prob <= 0:
    return 0.
  n_queries = data_split.num_queries()

  bounds = np.zeros(n_queries, dtype=np.float64)
  for qid in np.arange(n_queries):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    n_clicks = float(np.sum(doc_clicks[s_i:e_i]))
    min_prop = np.amin(doc_prop[s_i:e_i])
    b = max_reward/float(min_prop)
    if n_clicks > 1:
      bf_sqrt = np.log(2./(1.-conf_prob))*variance[qid]*(n_clicks/(n_clicks-1.))*n_clicks*2.
      right_part = np.sqrt(bf_sqrt)/n_clicks
      left_part = 7*b*np.log(2./(1.-conf_prob))/(3*(n_clicks-1))
      bounds[qid] = left_part + right_part
    else:
      bounds[qid] = np.inf
  return bounds

def confidence_bound_click_pattern_per_query(conf_prob, variance,
                                             max_reward, query_freq):
  if conf_prob <= 0:
    return 0.
  n_queries = variance.shape[0]

  bounds = np.zeros(n_queries, dtype=np.float64)
  for qid in np.arange(n_queries):
    n_samples = query_freq[qid]
    b = max_reward[qid]
    if n_samples > 1:
      bf_sqrt = np.log(2./(1.-conf_prob))*variance[qid]*(n_samples/(n_samples-1.))*n_samples*2.
      right_part = np.sqrt(bf_sqrt)/n_samples
      left_part = 7*b*np.log(2./(1.-conf_prob))/(3*(n_samples-1))
      bounds[qid] = left_part + right_part
    else:
      bounds[qid] = np.inf
  return bounds
