import os
from LModule import *
import torch
import numpy as np
from torch import nn
import faiss

# Adapted from code OpenOOD
# NOTE to self: post processing is something completely different from testing/predicting
class base_postprocessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _batches_forward(self,net, data):
        net = net.to(self.device)
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                output= net(batch)
            else:
                o = net(batch)
                output = torch.cat((output,o),0)
        return output

    def postprocess(self, net, data):
        output = self._batches_forward(net,data) 
        scores = torch.softmax(output, dim=1)
        conf, pred = torch.max(scores, dim=1)
        return pred, conf, scores


class dropout_postprocessor:
    def __init__(self, dropout_times = 10):
        self.dropout_times = dropout_times
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _forward_batches(self, net, data):
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                output = net.forward(batch)
            else:
                o = net.forward(batch)
                output = torch.cat((output,o),0) 
        return(output)

    def postprocess(self, net, data):
        net = net.to(self.device)
        net = net.train()  # to be in training mode  
        logits_list = [self._forward_batches(data) for i in range(self.dropout_times)] # pass in batches with the dtaaloader
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _batches_forward(self,net, data):
        net = net.to(self.device)
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                output= net(batch)
            else:
                o = net(batch)
                output = torch.cat((output,o),0)
        return output

    def postprocess(self, net, data):
        output = self._batches_forward(net,data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        return pred, conf, score

KNN_normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNN_postprocessor():
    def __init__(self, k):
        super(KNN_postprocessor, self).__init__()
        self.K = k
        self.setup_flag = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _batches_forward(self, net, data):
        net = net.to(self.device)
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                output, feature = net(batch, return_feature=True)
            else:
                o, f = net(batch, return_feature=True)
                feature = torch.cat((feature,f),0)
                output = torch.cat((output,o),0)
        return output, feature

    def setup(self, net, data):
        if not self.setup_flag:
            # 1) set up the index, based on the features
            _, feature = self._batches_forward(net, data)  # pass in batches with the dataloader
            d = feature.detach().cpu().numpy().shape[1]
            self.index = faiss.IndexFlatL2(d)

            # 2) add (normalized) vectors 
            self.index.add(KNN_normalizer(feature.detach().cpu().numpy()))

            self.setup_flag = True
        else:
            pass
            
    def postprocess(self, net, data):
        output, feature = self._net_with_feature(net, data) # pass in batches with the dataloader
        feature_normed = KNN_normalizer(feature.detach().cpu().numpy())
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
        self.num_networks = 10
        self.checkpoint_dirs = [
            name_analysis +'_' + str(i) + "_best_model.ckpt"
            for i in range(0, self.num_networks)
        ]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _batches_forward(self, net, data):
        net = net.to(self.device)
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                output= net(batch)
            else:
                o = net(batch)
                output = torch.cat((output,o),0)
        return output

    def setup(self):
        os.chdir(self.checkpoint_root)
        # Extract lightning modules
        self.modules= [
            LitBasicNN.load_from_checkpoint(self.checkpoint_dirs[i]) for i in range(0,self.num_networks)
        ]
        
        # Extract networks
        self.networks = [self.modules[i].NN for i in range(0, self.num_networks)]
        self.networks = [i.to(self.device) for i in self.networks]

    def postprocess(self, data):
        # Get output model
        logits_list = [
            self._batches_forward(self.networks[i],data)for i in range(self.num_networks)
        ]

        logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
        for i in range(self.num_networks):
            logits_mean += logits_list[i]
        logits_mean /= self.num_networks

        score = torch.softmax(logits_mean, dim=1) 
        conf, pred = torch.max(score, dim=1)
        return pred, conf

class Posterior_postprocessor(): 
    def __init__(self, uncertainty_type = "epistemic", loss = "UCE"):
        self.uncertainty_type = uncertainty_type
        self.loss = loss
        if torch.cuda.is_available():
            self.dev = 'cuda' 
        else: 
            self.dev = 'cpu'

    def _batches_forward(self,net, data):
        net = net.to(self.device)
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                alpha, soft_output_pred = net(batch)
            else:
                a, s = net(batch)
                alpha = torch.cat((alpha, a),0)
                soft_output_pred = torch.cat((soft_output_pred,s),0)
        return alpha, soft_output_pred

    def postprocess(self, net, data):
        data = data.to(self.dev)
        if self.loss == "UCE":
            alpha, soft_output_pred = self._batches_forward(net,data)
        else:
            NotImplementedError

        if self.uncertainty_type == 'epistemic':
            scores = alpha.sum(-1).cpu().detach().numpy()
        elif self.uncertainty_type == 'aleatoric':
            p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
            scores = p.max(-1)[0].cpu().detach().numpy()
   
        _, pred = torch.max(soft_output_pred, dim=1) 
        return pred, scores


