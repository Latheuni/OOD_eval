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
                o = net(batch)
                output = o.detach().to('cpu')
            else:
                o = net(batch)
                o = o.detach().to('cpu')
                output = torch.cat((output,o),0)
            del batch
            torch.cuda.empty_cache()
        return output

    def postprocess(self, net, data):
        output = self._batches_forward(net,data) 
        scores = torch.softmax(output, dim=1)
        conf, pred = torch.max(scores, dim=1)
        return pred, conf, scores


class dropout_postprocessor:
    def __init__(self, dropout_times = 10, mode = "normal"):
        self.dropout_times = dropout_times
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mode = mode

    def _forward_batches(self, net, data):
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                o = net.forward(batch)
                output = o.detach().to('cpu')
            else:
                o = net.forward(batch)
                o = o.detach().to('cpu')
                output = torch.cat((output,o),0) 
            del batch
            torch.cuda.empty_cache()
        return(output)
    def postprocess(self, net, data):
        if self.mode == "normal":
            net = net.to(self.device)
            net = net.train()  # to be in training mode  
            logits_list = [self._forward_batches(net,data) for i in range(self.dropout_times)] # pass in batches with the dataloader
            logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
            for i in range(self.dropout_times):
                logits_mean += logits_list[i]
            logits_mean /= self.dropout_times
            score = torch.softmax(logits_mean, dim=1)
            conf, pred = torch.max(score, dim=1)
            return pred, conf, score
        elif self.mode == "variance":
            net = net.to(self.device)
            net = net.train()  # to be in training mode  
            logits_list = [self._forward_batches(net,data) for i in range(self.dropout_times)] # pass in batches with the dataloader

            # calculate predictions
            logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
            for i in range(self.dropout_times):
                logits_mean += logits_list[i]
            logits_mean /= self.dropout_times
            score = torch.softmax(logits_mean, dim=1)
            _, pred = torch.max(score, dim=1)

            # calculate conf based on variance of logits
            ## perform softmax on the logits
            logits_list_softmax = []
            
            for i in range(self.dropout_times):
                logits_list_softmax.append(torch.softmax(logits_list[i], dim=1))
            #input = torch.empty(list(logits_list_softmax[0].shape)[0])
            logits_preds = []
            
            for n in range(len(pred)):
                n_ = []
                for i in range(self.dropout_times):
                    
                        if i == 0:
                            n_ = [logits_list_softmax[i][n][pred[n]]]
                        else:
                            n_.append(logits_list_softmax[i][n][pred[n]])
                logits_preds.append(n_)
            var = np.var(logits_preds, axis=1)
            conf = 1-var
            return pred, conf, score
        else:
            raise ValueError("mode argument is invalid, needs to be empty, str(normal) or str(variance)")


class EBO_postprocessor:
    def __init__(self, temperature=1):
        self.temperature = temperature
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def _batches_forward(self,net, data):
        net = net.to(self.device)
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.device)
            if i == 0:
                o = net(batch)
                output = o.detach().to('cpu')
            else:
                o = net(batch)
                o = o.detach().to('cpu')
                output = torch.cat((output,o),0)
            del batch
            torch.cuda.empty_cache()
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
                o, f = net(batch, return_feature=True)
                output = o.detach().to('cpu')
                feature = f.detach().to('cpu')
            else:
                o, f = net(batch, return_feature=True)
                o = o.detach().to('cpu')
                f = f.detach().to('cpu')
                feature = torch.cat((feature,f),0) 
                output = torch.cat((output,o),0)
            del batch
            torch.cuda.empty_cache()
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
        output, feature = self._batches_forward(net, data) # pass in batches with the dataloader
        feature_normed = KNN_normalizer(feature.detach().cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)


class Ensemble_postprocessor():
    def __init__(self, model_dir, name_analysis, mode = "normal"):
        super(Ensemble_postprocessor, self).__init__()
        
        self.checkpoint_root = model_dir
        self.mode = mode
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
                o = net(batch)
                output = o.detach().to('cpu')
            else:
                o = net(batch)
                o = o.detach().to('cpu')
                output = torch.cat((output,o),0)
            del batch
            torch.cuda.empty_cache()
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
        if self.mode == "normal":
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
            return pred, conf, score
        elif self.mode == "variance":
            logits_list = [
                self._batches_forward(self.networks[i],data)for i in range(self.num_networks)
            ]

            # calculate predictions
            logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
            for i in range(self.dropout_times):
                logits_mean += logits_list[i]
            logits_mean /= self.dropout_times
            score = torch.softmax(logits_mean, dim=1)
            _, pred = torch.max(score, dim=1)

            # calculate conf based on variance of logits
            ## perform softmax on the logits
            logits_list_softmax = []
            
            for i in range(len(self.networks)):
                logits_list_softmax.append(torch.softmax(logits_list[i], dim=1))
            #input = torch.empty(list(logits_list_softmax[0].shape)[0])
            logits_preds = []
            
            for n in range(len(pred)):
                n_ = []
                for i in range(len(self.networks)):
                        if i == 0:
                            n_ = [logits_list_softmax[i][n][pred[n]]]
                        else:
                            n_.append(logits_list_softmax[i][n][pred[n]])
                logits_preds.append(n_)
            var = np.var(logits_preds, axis=1)
            conf = 1-var
            return pred, conf, score
        else:
            raise ValueError("mode argument is invalid, needs to be empty, str(normal) or str(variance)")

class Posterior_postprocessor(): 
    def __init__(self, uncertainty_type = "epistemic", loss = "UCE"):
        self.uncertainty_type = uncertainty_type
        self.loss = loss
        if torch.cuda.is_available():
            self.dev = 'cuda' 
        else: 
            self.dev = 'cpu'

    def _batches_forward(self,net, data):
        net = net.to(self.dev)
        for i, (batch, y) in enumerate(data):
            batch = batch.to(self.dev)
            if i == 0:
                a, s = net(batch)
                alpha = a.detach().to('cpu')
                soft_output_pred = s.detach().to('cpu')
            else:
                a, s = net(batch)
                a = a.detach().to('cpu')
                s = s.detach().to('cpu')
                alpha = torch.cat((alpha, a),0)
                soft_output_pred = torch.cat((soft_output_pred,s),0)
            del batch
            torch.cuda.empty_cache()
        return alpha, soft_output_pred

    def postprocess(self, net, data):
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


