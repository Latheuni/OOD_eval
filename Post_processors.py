import os
from LModule import *
import torch
import numpy as np
from torch import nn
# import faiss

# Adapted from code OpenOOD
# NOTE to self: post processing is something completely different from testing/predicting
class base_postprocessor:
    def postprocess(self, net, data):
        output = net(data)
        scores = torch.softmax(output, dim=1)
        conf, pred = torch.max(scores, dim=1)
        return pred, conf, scores


class dropout_postprocessor:
    def __init__(self, dropout_times = 10):
        self.dropout_times = dropout_times

    def postprocess(self, net, data):
        net = net.train()  # to be in training mode
        logits_list = [net.forward(data) for i in range(self.dropout_times)]
        logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
        for i in range(self.dropout_times):
            logits_mean += logits_list[i]
        logits_mean /= self.dropout_times
        score = torch.softmax(logits_mean, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf, score


class EBO_postprocessor:
    def __init__(self, temperature=1):
        self.temperature = temperature

    def postprocess(self, net, data):
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        return pred, conf, score

KNN_normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNNPostprocessor():
    def __init__(self, k):
        super(KNNPostprocessor, self).__init__(config)
        self.K = k
        self.setup_flag = False
    
    def setup(self, data):
        if not self.setup_flag:
            # 1) set up the index, based on the features
              _, feature = net(data, return_feature=True) 
            d = feature.cpu().numpy().shape[1]
            print('check KNN post d', d) 
            self.index = faiss.IndexFlatL2(d)

            # 2) add (normalized) vectors 
            self. index.add(normalizer(data))

            self.setup_flag = True
        else:
            pass
            
    def postprocess(self, net, data):
        output, feature = net(data, return_feature=True) 
        print("Check KNN postprocessor", feature.shape)
        feature_normed = normalizer(feature.cpu().numpy())
        print("Check KNN postprocessor normalizer", feature_normed.shape)
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)


class Ensemble_postprocessor():
    def __init__(self, model_dir, name_analysis):
        super(Ensemble_postprocessor, self).__init__()
        
        self.checkpoint_root = model_dir

        # number of networks to esembel
        self.num_networks = 10
        # get networks TODO: carefull: arbitrary naming here
        self.checkpoint_dirs = [
            name_analysis +'_' + str(i) + "_best_model.ckpt"
            for i in range(0, self.num_networks)
        ]

    # TODO restructure to load in the models
    def setup(self, net):
        os.chdir(self.checkpoint_root)
        # Extract lightning modules
        self.modules= [
            LitBasicNN.load_from_checkpoint(self.checkpoint_dirs[i]) for i in range(0,self.num_networks)
        ]

        # Extract networks
        self.networks = [self.modules[i].NN for i in range(0, self.num_networks)]

    def postprocess(self, data):
        # Get output model
        logits_list = [
            self.networks[i](data) for i in range(self.num_networks)
        ]

        #         logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
        for i in range(self.num_networks):
            logits_mean += logits_list[i]
        logits_mean /= self.num_networks

        score = torch.softmax(logits_mean, dim=1) #TODO: Q Thomas
        conf, pred = torch.max(score, dim=1)
        return pred, conf