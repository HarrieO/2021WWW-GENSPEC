# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np

def read_model(model_file_path, data, scale=1.0):
  model = np.zeros(data.num_features)
  with open(model_file_path, 'r') as model_file:
    model_line = model_file.readlines()[-1]
    model_line = model_line[:model_line.find('#')]
    for feat_tuple in model_line.split()[1:]:
      f_i, f_v = feat_tuple.split(':')
      f_i = data.inverse_feature_map[int(f_i)]
      model[f_i] = float(f_v)
  model /= np.linalg.norm(model)/scale
  return model
