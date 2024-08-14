import numpy as np
import torch
import os

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser


class Explainer():

    def __init__(self):
        self.task='traffic_state_pred'
        self.model_name='STGCN'
        self.dataset_name='METR_LA'
        self.config_file=None
        self.saved_model=True
        self.train=False
        self.other_args={'exp_id': '1', 'seed': 0}
        self.config = ConfigParser(self.task, self.model_name, self.dataset_name, self.config_file, self.saved_model, self.train, self.other_args)
        self.device = self.config.get('device', torch.device('cpu'))



    def load_data(self):

        dataset = get_dataset(self.config)
        train_data, valid_data, test_data = dataset.get_data()
        data_feature = dataset.get_data_feature()
        return data_feature, train_data, valid_data, test_data

    def load_model(self, model_path, data_feature):

        model = get_model(self.config, data_feature)
        model_state, optimizer_state = torch.load(model_path)
        model.load_state_dict(model_state)
        model = model.to(self.device)

        return model

    def main(self):

        data_feature, train_data, valid_data, test_data = self.load_data()
        model_path = os.getcwd()+'/libcity/cache/1/model_cache/STGCN_METR_LA.m'
        model = self.load_model(model_path, data_feature)

        with torch.no_grad():
            model.eval()
            for batch in test_data:
                x = batch['X']
                print('X Shape: ',np.array(x).shape)
                y = batch['y']
                print('Y Shape: ',np.array(y).shape)
                batch.to_tensor(self.device)
                output = model.predict(batch)
                print(output.shape)

    def fetch_computation_graph(self, num_layers=2):


        return output


exp = Explainer()
output = exp.main()
