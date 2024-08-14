import numpy as np
import torch
import os

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser

import warnings
warnings.filterwarnings("ignore")

# If we save some batches of input data can we laod in a model and make predictions



task='traffic_state_pred'
model_name='STGCN'
dataset_name='METR_LA'
config_file=None
saved_model=True
train=False
other_args={'exp_id': '1', 'seed': 0}
config = ConfigParser(task, model_name, dataset_name, config_file, saved_model, train, other_args)

device = config.get('device', torch.device('cpu'))

dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()

data_feature = dataset.get_data_feature()

model = get_model(config, data_feature)
model_path = os.getcwd()+'/libcity/cache/1/model_cache/STGCN_METR_LA.m'
model_state, optimizer_state = torch.load(model_path)
model.load_state_dict(model_state)
model = model.to(device)

with torch.no_grad():
    model.eval()
    for batch in test_data:
        x = batch['X']
        y = batch['y']
        batch.to_tensor(device)
        output = model.predict(batch)
