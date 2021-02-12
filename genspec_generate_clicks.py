# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import pickle
import multiprocessing
import random
from multiprocessing import Pool

import utils.dataset as dataset
import utils.pretrained_models as prtr
import utils.clicks as clk
import utils.ranking as rnk

parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str,
                    help="Model file output from pretrained model.")
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
parser.add_argument("--scale", type=float,
                    default=5.0,
                    help="Scaling the linear scorer.")
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
num_proc = args.num_proc
assert num_proc >= 0, 'Invalid number of processes: %d' % num_proc
perc = args.perc

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

train_query_probs = None
validation_query_probs = None

pretrain_model = prtr.read_model(args.model_file, data, args.scale)

train_ranking = rnk.data_split_model_rank_and_invert(
                      pretrain_model,
                      data.train
                    )[1]
validation_ranking = rnk.data_split_model_rank_and_invert(
                            pretrain_model,
                            data.validation
                          )[1]

def _generate_clicks(m_args):
  split_name, num_clicks, click_patterns = m_args
  seed = int(multiprocessing.current_process()._identity[0]*(num_clicks%56789))
  random.seed((time.time(), seed))
  np.random.seed(int(time.time() + seed * 100 + seed))
  if split_name == 'train':
    data_split = data.train
    cur_ranking = train_ranking
  elif split_name == 'validation':
    data_split = data.validation
    cur_ranking = validation_ranking
  return clk.generate_clicks(
                    data_split,
                    cur_ranking,
                    click_model,
                    num_clicks,
                    eta,
                    click_patterns=click_patterns,
                    )

if num_proc > 1:
  train_ranking = clk._make_shared(train_ranking)
  validation_ranking = clk._make_shared(validation_ranking)
  pool = Pool(processes=args.num_proc)

print('About to generate clicks')
start = time.time()

click_list = [10**x for x in range(2, 10)]
click_list += [2*10**x for x in range(2, 9)]
click_list += [5*10**x for x in range(2, 9)]
click_list = sorted(click_list)

result = None
for num_clicks in click_list:
  num_train_clicks = int(num_clicks*(1.-perc))
  num_select_clicks = int(num_clicks*perc)
  num_validation_clicks = int(num_clicks*0.15)
  if result:
    num_train_clicks -= result['train']['num_clicks']
    num_select_clicks -= result['select']['num_clicks']
    num_validation_clicks -= result['validation']['num_clicks']
  if num_proc <= 1 or num_clicks <= 10**6:
    train_clicks = clk.generate_clicks(
                        data.train,
                        train_ranking,
                        click_model,
                        num_train_clicks,
                        eta)
    select_clicks = clk.generate_clicks(
                        data.train,
                        train_ranking,
                        click_model,
                        num_select_clicks,
                        eta,
                        click_patterns=False,
                        )
    validation_clicks = clk.generate_clicks(
                              data.validation,
                              validation_ranking,
                              click_model,
                              num_validation_clicks,
                              eta)
  else:
    job_args = [(data.train.name, int(num_train_clicks/num_proc), False)] * num_proc
    clicks_left = np.sum([x[1] for x in job_args[1:]])
    job_args[0] = (data.train.name,
                   int(num_train_clicks - clicks_left),
                   False)
    train_sub_clicks = pool.map(_generate_clicks, job_args)
    train_clicks = clk.add_clicks(train_sub_clicks)

    job_args = [(data.train.name, int(num_select_clicks/num_proc), True)] * num_proc
    clicks_left = np.sum([x[1] for x in job_args[1:]])
    job_args[0] = (data.train.name,
                   int(num_select_clicks - clicks_left),
                   True)
    select_sub_clicks = pool.map(_generate_clicks, job_args)
    select_clicks = clk.add_clicks(select_sub_clicks)

    job_args = [(data.validation.name, int(num_validation_clicks/num_proc), False)] * num_proc
    clicks_left = np.sum([x[1] for x in job_args[1:]])
    job_args[0] = (data.validation.name,
                   int(num_validation_clicks - clicks_left),
                   False)
    validation_sub_clicks = pool.map(_generate_clicks, job_args)
    validation_clicks = clk.add_clicks(validation_sub_clicks)

  if result is None:
    result = {
        'train': train_clicks,
        'select': select_clicks,
        'validation': validation_clicks,
        'click_model': click_model,
        'eta': eta, 
        'dataset_name': data.name,
        'dataset_fold': data.fold_num,
        'target_num_clicks': num_clicks,
      }
  else:
    result['train'] = clk.add_clicks([result['train'],
                                      train_clicks])
    result['select'] = clk.add_clicks([result['select'],
                                       select_clicks])
    result['validation'] = clk.add_clicks([result['validation'],
                                           validation_clicks])
    result['target_num_clicks'] = num_clicks

  file_path = args.click_path + '%sclicks.pkl' % num_clicks
  print('Saving result to:', file_path)
  with open(file_path, 'wb') as f:
    pickle.dump(result, f, protocol=4)
  print('Clicks per second:',
        (result['train']['num_clicks']
         + result['select']['num_clicks']
         + result['validation']['num_clicks']
         )/float(time.time() - start))
