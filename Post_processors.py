import torch
import numpy as np
from torch import nn


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


# class EnsemblePostprocessor(BasePostprocessor):
#     def __init__(self, config):
#         super(EnsemblePostprocessor, self).__init__(config)
#         self.config = config
#         self.postprocess_config = config.postprocessor
#         self.postprocessor_args = self.postprocess_config.postprocessor_args
#         assert self.postprocessor_args.network_name == \
#             self.config.network.name,\
#             'checkpoint network type and model type do not align!'
#         # get ensemble args
#         self.checkpoint_root = self.postprocessor_args.checkpoint_root

#         # list of trained network checkpoints
#         self.checkpoints = self.postprocessor_args.checkpoints
#         # number of networks to esembel
#         self.num_networks = self.postprocessor_args.num_networks
#         # get networks
#         self.checkpoint_dirs = [
#             osp.join(self.checkpoint_root, path, 'best.ckpt')
#             for path in self.checkpoints
#         ]

#     def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
#         self.networks = [deepcopy(net) for i in range(self.num_networks)]
#         for i in range(self.num_networks):
#             self.networks[i].load_state_dict(torch.load(
#                 self.checkpoint_dirs[i]),
#                                              strict=False)
#             self.networks[i].eval()

#     def postprocess(self, net: nn.Module, data: Any):
#         logits_list = [
#             self.networks[i](data) for i in range(self.num_networks)
#         ]
#         logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
#         for i in range(self.num_networks):
#             logits_mean += logits_list[i]
#         logits_mean /= self.num_networks

#         score = torch.softmax(logits_mean, dim=1)
#         conf, pred = torch.max(score, dim=1)
#         return pred, conf


# normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
# class KNN_postprocessor():
#     def __init__(self, K):
#         self.K = K

#     def setup(self):
#  with torch.no_grad():
#             for batch in tqdm(id_loader_dict['train'],
#                               desc='Setup: ',
#                               position=0,
#                               leave=True):
#                 data = batch['data'].cuda()
#                 data = data.float()

#                 _, feature = net(data, return_feature=True)
#                 activation_log.append(
#                     normalizer(feature.data.cpu().numpy()))

#         self.activation_log = np.concatenate(activation_log, axis=0)
#         self.index = faiss.IndexFlatL2(feature.shape[1])
#         self.index.add(self.activation_log)

# def postprocess(self):
# need to add a return feature
# feature_normed = normalizer(feature.data.cpu().numpy())
# postprocessor_args:

#  D, _ = self.index.search(
#     feature_normed,
#     self.K,
# )
# kth_dist = -D[:, -1]
# _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
# return pred, torch.from_numpy(kth_dist)


# K: 50
# postprocessor_sweep:
#  K_list: [50, 100, 200, 500, 1000]
