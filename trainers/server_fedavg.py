import glob
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import json
import math
import torch
import random
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore
from typing import Optional, List, OrderedDict, Tuple, Dict
from flwr.common import Parameters, Scalar, MetricsAggregationFn, FitIns, EvaluateIns, FitRes, EvaluateRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from models.stackoverflow_nwp_rnn import StackoverflowNet as StackoverflowNWPNet
from models.stackoverflow_nwp_rnn_memoization import StackoverflowNet as StackoverflowNWPNetMemo
from models.stackoverflow_nwp_rnn_per_client_blocks import StackoverflowNet as StackoverflowNWPNetBlock
from models.stackoverflow_nwp_rnn_dynamic_v4 import StackoverflowNet as StackoverflowNWPNetDynamicV4
from models.stackoverflow_nwp_rnn_dynamic_v6 import StackoverflowNet as StackoverflowNWPNetDynamicV6
from models.synthetic_fc import SyntheticNet
from models.emnist_cnn import EMNISTNet
from models.emnist_cnn_memoization import EMNISTNet as EMNISTNetMemo
from models.emnist_cnn_per_client_blocks import EMNISTNet as EMNISTNetBlock
from models.emnist_cnn_dynamic_v1 import EMNISTNet as EMNISTNetDynamicV1
from models.emnist_cnn_dynamic_v4 import EMNISTNet as EMNISTNetDynamicV4
from models.emnist_cnn_dynamic_v6 import EMNISTNet as EMNISTNetDynamicV6
from models.cifar100_resnet import resnet18 as CifarResNet
from models.cifar100_resnet_memoization import resnet18 as CifarResNetMemo
from models.cifar100_resnet_per_client_blocks import resnet18 as CifarResNetBlock
from models.cifar100_resnet_dynamic_v4 import resnet18 as CifarResNetDynamicV4
from models.cifar100_resnet_dynamic_v6 import resnet18 as CifarResNetDynamicV6
from models.shakespeare_rnn import ShakespeareNet
from models.shakespeare_rnn_memoization import ShakespeareNet as ShakespeareNetMemo
from models.shakespeare_rnn_per_client_blocks import ShakespeareNet as ShakespeareNetBlock
from models.shakespeare_rnn_dynamic_v4 import ShakespeareNet as ShakespeareNetDynamicV4
from models.shakespeare_rnn_dynamic_v6 import ShakespeareNet as ShakespeareNetDynamicV6
from models.stackoverflow_tag_lr import StackoverflowLogisticRegression as StackoverflowLRNet

from configs.hyperparameters import *

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class FedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        dataset: str = 'stackoverflow',
        client_algorithm: str = 'FedAvg',
        shift_ratio: float = 1.00,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.dataset_name = dataset
        self.dataset_hyperparameters = eval(str(dataset)+'_'+str(client_algorithm).lower())
        self.client_algorithm = client_algorithm

        self.lr = self.dataset_hyperparameters['lr']
        self.epochs = self.dataset_hyperparameters['epochs']
        self.total_rounds = self.dataset_hyperparameters['rounds']
        self.num_clients = self.dataset_hyperparameters['num_clients']
        self.total_clients = self.dataset_hyperparameters['total_clients']
        self.checkpoint_dir = self.dataset_hyperparameters['checkpoint_dir']
        self.checkpoint_interval = self.dataset_hyperparameters['checkpoint_interval']

        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.model_file_name = self.checkpoint_dir + 'model_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.pth'
        self.loss_file_name = self.checkpoint_dir + 'losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.before_accu_file_name = self.checkpoint_dir + 'before_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.after_accu_file_name = self.checkpoint_dir + 'after_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        
        self.all_loss_file_name = self.checkpoint_dir + 'all_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.all_before_accu_file_name = self.checkpoint_dir + 'all_before_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.all_after_accu_file_name = self.checkpoint_dir + 'all_after_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        
        self.plot_loss_file_name = self.checkpoint_dir + 'plot_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        self.plot_before_accu_file_name = self.checkpoint_dir + 'plot_before_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        self.plot_after_accu_file_name = self.checkpoint_dir + 'plot_after_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        
        self.personalized_accu_file_name = self.checkpoint_dir + 'personalized_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.personalized_all_accu_file_name = self.checkpoint_dir + 'all_personalized_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.personalized_plot_accu_file_name = self.checkpoint_dir + 'plot_personalized_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        
        self.w_g_true_w_p_true_file_name = self.checkpoint_dir + 'w_g_true_w_p_true_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.w_g_true_w_p_false_file_name = self.checkpoint_dir + 'w_g_true_w_p_false_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.w_g_false_w_p_true_file_name = self.checkpoint_dir + 'w_g_false_w_p_true_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.w_g_false_w_p_false_file_name = self.checkpoint_dir + 'w_g_false_w_p_false_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.harmed_file_name = self.checkpoint_dir + 'harmed_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.sample_count_file_name = self.checkpoint_dir + 'sample_count_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        
        self.highest_sampled_personalized_accuracy_file_name = self.checkpoint_dir + 'highest_sampled_personalized_accuracy_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.lowest_sampled_personalized_accuracy_file_name = self.checkpoint_dir + 'lowest_sampled_personalized_accuracy_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'

        self.losses_dict = {}
        self.before_accuracies_dict = {}
        self.after_accuracies_dict = {}
        self.all_losses_dict = {}
        self.all_before_accuracies_dict = {}
        self.all_after_accuracies_dict = {}
        self.personalized_accuracies_dict = {}
        self.all_personalized_accuracies_dict = {}

        self.w_g_true_w_p_true_dict = {}
        self.w_g_true_w_p_false_dict = {}
        self.w_g_false_w_p_true_dict = {} 
        self.w_g_false_w_p_false_dict = {}
        self.harmed_dict = {}
        self.sample_count_dict = {}

        self.highest_sampled_personalized_accuracy_dict = {}
        self.lowest_sampled_personalized_accuracy_dict = {}
        
        self.round_offset = 0

        try:
            os.remove(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt')
        except:
            pass
        
        if os.path.exists(self.loss_file_name):
            with open(self.loss_file_name, 'r') as f:
                data_raw = f.read()
            self.losses_dict = json.loads(data_raw)

            with open(self.all_loss_file_name, 'r') as f:
                data_raw = f.read()
            self.all_losses_dict = json.loads(data_raw)

            with open(self.before_accu_file_name, 'r') as f:
                data_raw = f.read()
            self.before_accuracies_dict = json.loads(data_raw)
            
            with open(self.after_accu_file_name, 'r') as f:
                data_raw = f.read()
            self.after_accuracies_dict = json.loads(data_raw)

            with open(self.all_before_accu_file_name, 'r') as f:
                data_raw = f.read()
            self.all_before_accuracies_dict = json.loads(data_raw)

            with open(self.all_after_accu_file_name, 'r') as f:
                data_raw = f.read()
            self.all_after_accuracies_dict = json.loads(data_raw)

            with open(self.personalized_accu_file_name, 'r') as f:
                data_raw = f.read()
            self.personalized_accuracies_dict = json.loads(data_raw)

            with open(self.personalized_all_accu_file_name, 'r') as f:
                data_raw = f.read()
            self.all_personalized_accuracies_dict = json.loads(data_raw)

            with open(self.w_g_true_w_p_true_file_name, 'r') as f:
                data_raw = f.read()
            self.w_g_true_w_p_true_dict = json.loads(data_raw)

            with open(self.w_g_true_w_p_false_file_name, 'r') as f:
                data_raw = f.read()
            self.w_g_true_w_p_false_dict = json.loads(data_raw)

            with open(self.w_g_false_w_p_true_file_name, 'r') as f:
                data_raw = f.read()
            self.w_g_false_w_p_true_dict = json.loads(data_raw)

            with open(self.w_g_false_w_p_false_file_name, 'r') as f:
                data_raw = f.read()
            self.w_g_false_w_p_false_dict = json.loads(data_raw)

            with open(self.harmed_file_name, 'r') as f:
                data_raw = f.read()
            self.harmed_dict = json.loads(data_raw)

            with open(self.sample_count_file_name, 'r') as f:
                data_raw = f.read()
            self.sample_count_dict = json.loads(data_raw)
            
            with open(self.highest_sampled_personalized_accuracy_file_name, 'r') as f:
                data_raw = f.read()
            self.highest_sampled_personalized_accuracy_dict = json.loads(data_raw)
            
            with open(self.lowest_sampled_personalized_accuracy_file_name, 'r') as f:
                data_raw = f.read()
            self.lowest_sampled_personalized_accuracy_dict = json.loads(data_raw)

            self.round_offset = max([int(i) for i in self.losses_dict.keys()])

        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_eval, min_fit_clients=min_fit_clients, min_evaluate_clients=min_eval_clients, min_available_clients=min_available_clients, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):# -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        client_proxy_and_fitins = super().configure_fit(server_round, parameters, client_manager)
        if self.dataset_name == 'stackoverflow_nwp': 
            client_proxy_and_fitins[0][0].cid = '01165998'
            client_proxy_and_fitins[1][0].cid = '03292394'
        elif self.dataset_name == 'emnist': 
            client_proxy_and_fitins[0][0].cid = 'f0289_10'
            client_proxy_and_fitins[1][0].cid = 'f1727_36'

        for idx, i in enumerate(client_proxy_and_fitins):
            i[1].config['rnd'] = server_round
            
        return client_proxy_and_fitins

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):# -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        client_proxy_and_fitins = super().configure_evaluate(server_round, parameters, client_manager)
        if self.dataset_name == 'stackoverflow_nwp': 
            client_proxy_and_fitins[0][0].cid = '01165998'
            client_proxy_and_fitins[1][0].cid = '03292394'
        elif self.dataset_name == 'emnist': 
            client_proxy_and_fitins[0][0].cid = 'f0289_10'
            client_proxy_and_fitins[1][0].cid = 'f1727_36'

        for idx, i in enumerate(client_proxy_and_fitins):
            i[1].config['rnd'] = server_round

        with open(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt') as f:
            accuracies = f.readlines()
            accuracies = [float(acc.strip()) for acc in accuracies]
            
            accuracy_aggregated = sum(accuracies) / len(accuracies)
            print(Fore.RED + f"{self.client_algorithm}: Round {server_round + self.round_offset} personalized accuracy aggregated from client results: {accuracy_aggregated}" + Fore.WHITE)
            os.remove(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt')
            
            self.personalized_accuracies_dict[int(server_round + self.round_offset)] = accuracy_aggregated
            self.all_personalized_accuracies_dict[int(server_round + self.round_offset)] = accuracies

        if (server_round + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.personalized_accu_file_name, 'w') as f:
                json.dump(self.personalized_accuracies_dict, f)

            with open(self.personalized_all_accu_file_name, 'w') as f:
                json.dump(self.all_personalized_accuracies_dict, f)

            plt.figure(0)
            plt.plot(self.personalized_accuracies_dict.keys(), self.personalized_accuracies_dict.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized Accuracy')
            plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.personalized_plot_accu_file_name)

        return client_proxy_and_fitins

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):# -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        loss_aggregated = sum(losses) / sum(examples)
        print("")
        print(Fore.BLUE + f"{self.client_algorithm}: Round {rnd + self.round_offset} loss aggregated from client results: {loss_aggregated}" + Fore.WHITE)

        self.losses_dict[int(rnd + self.round_offset)] = loss_aggregated

        all_losses = [r.metrics["loss"] for _, r in results]
        self.all_losses_dict[int(rnd + self.round_offset)] = all_losses

        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.loss_file_name, 'w') as f:
                json.dump(self.losses_dict, f)

            with open(self.all_loss_file_name, 'w') as f:
                json.dump(self.all_losses_dict, f)

            plt.figure(1)
            plt.plot(self.losses_dict.keys(), self.losses_dict.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Training Loss')
            plt.title('Training Loss for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_loss_file_name)

            print(Fore.GREEN + f"{self.client_algorithm}: Saving aggregated weights at round {rnd + self.round_offset}" + Fore.WHITE)
            
            if self.dataset_name == 'stackoverflow_nwp':
                vocab_size = self.dataset_hyperparameters['vocab_size'] + 4
                embedding_size = self.dataset_hyperparameters['embedding_size']
                hidden_size = self.dataset_hyperparameters['hidden_size']
                num_layers = self.dataset_hyperparameters['num_layers']
                sequence_length = self.dataset_hyperparameters['sequence_length']

                if self.client_algorithm == 'knnPer':
                    net = StackoverflowNWPNetMemo(vocab_size, embedding_size, hidden_size, num_layers)
                elif self.client_algorithm == 'PartialFed':
                    net = StackoverflowNWPNetBlock(vocab_size, embedding_size, hidden_size, num_layers)
                elif self.client_algorithm in ['FlowV2', 'FlowV3', 'FlowV4', 'FlowV5']:
                    net = StackoverflowNWPNetDynamicV4(vocab_size, embedding_size, hidden_size, sequence_length, num_layers)
                elif self.client_algorithm in ['FlowV6']:
                    net = StackoverflowNWPNetDynamicV6(vocab_size, embedding_size, hidden_size, sequence_length, num_layers)
                elif self.client_algorithm == 'HypCluster':
                    net_1 = StackoverflowNWPNet(vocab_size, embedding_size, hidden_size, num_layers)
                    net_2 = StackoverflowNWPNet(vocab_size, embedding_size, hidden_size, num_layers)
                else:
                    net = StackoverflowNWPNet(vocab_size, embedding_size, hidden_size, num_layers)

            elif self.dataset_name == 'synthetic':
                input_dim = self.dataset_hyperparameters['input_dim']
                hidden_dim = self.dataset_hyperparameters['hidden_dim']
                output_dim = self.dataset_hyperparameters['output_dim']
                net = SyntheticNet(input_dim, hidden_dim, output_dim)

            elif self.dataset_name == 'emnist':
                if self.client_algorithm == 'knnPer':
                    net = EMNISTNetMemo()
                elif self.client_algorithm == 'PartialFed':
                    net = EMNISTNetBlock()
                elif self.client_algorithm == 'FlowV1':
                    batch_size = self.dataset_hyperparameters['batch_size']
                    net = EMNISTNetDynamicV1(batch_size=batch_size)
                elif self.client_algorithm in ['FlowV2', 'FlowV3', 'FlowV4', 'FlowV5',]:
                    net = EMNISTNetDynamicV4()
                elif self.client_algorithm in ['FlowV6']:
                    net = EMNISTNetDynamicV6()
                elif self.client_algorithm == 'HypCluster':
                    net_1 = EMNISTNet()
                    net_2 = EMNISTNet()
                else:
                    net = EMNISTNet()

            elif self.dataset_name == 'cifar10':
                if self.client_algorithm == 'knnPer':
                    net = CifarResNetMemo(num_classes=10)
                elif self.client_algorithm == 'PartialFed':
                    net = CifarResNetBlock(num_classes=10)
                elif self.client_algorithm in ['FlowV2', 'FlowV3', 'FlowV4', 'FlowV5',]:
                    net = CifarResNetDynamicV4(num_classes=10)
                elif self.client_algorithm in ['FlowV6']:
                    net = CifarResNetDynamicV6(num_classes=10)
                elif self.client_algorithm == 'HypCluster':
                    net_1 = CifarResNet(num_classes=10)
                    net_2 = CifarResNet(num_classes=10)
                else:
                    net = CifarResNet(num_classes=10)

            elif self.dataset_name == 'cifar100':
                if self.client_algorithm == 'knnPer':
                    net = CifarResNetMemo(num_classes=100)
                elif self.client_algorithm == 'PartialFed':
                    net = CifarResNetBlock(num_classes=100)
                elif self.client_algorithm in ['FlowV2', 'FlowV3', 'FlowV4', 'FlowV5',]:
                    net = CifarResNetDynamicV4(num_classes=100)
                elif self.client_algorithm in ['FlowV6']:
                    net = CifarResNetDynamicV6(num_classes=100)
                elif self.client_algorithm == 'HypCluster':
                    net_1 = CifarResNet(num_classes=100)
                    net_2 = CifarResNet(num_classes=100)
                else:
                    net = CifarResNet(num_classes=100)

            elif self.dataset_name == 'shakespeare':
                vocab_size = self.dataset_hyperparameters['vocab_size']
                embedding_size = self.dataset_hyperparameters['embedding_size']
                hidden_size = self.dataset_hyperparameters['hidden_size']
                if self.client_algorithm == 'knnPer':
                    net = ShakespeareNetMemo(embedding_size, vocab_size, hidden_size)
                elif self.client_algorithm == 'PartialFed':
                    net = ShakespeareNetBlock(embedding_size, vocab_size, hidden_size)
                elif self.client_algorithm in ['FlowV2', 'FlowV3', 'FlowV4', 'FlowV5',]:
                    net = ShakespeareNetDynamicV4(embedding_size, vocab_size, hidden_size)
                elif self.client_algorithm in ['FlowV6']:
                    net = ShakespeareNetDynamicV6(embedding_size, vocab_size, hidden_size)
                elif self.client_algorithm == 'HypCluster':
                    net_1 = ShakespeareNet(embedding_size, vocab_size, hidden_size)
                    net_2 = ShakespeareNet(embedding_size, vocab_size, hidden_size)
                else:
                    net = ShakespeareNet(embedding_size, vocab_size, hidden_size)

            elif self.dataset_name == 'stackoverflow_lr':
                word_vocab_size = self.dataset_hyperparameters['word_vocab_size']
                tag_vocab_size = self.dataset_hyperparameters['tag_vocab_size']
                net = StackoverflowLRNet(word_vocab_size, tag_vocab_size)

            if self.client_algorithm == 'HypCluster':
                aggregated_weights_1 = aggregated_weights[:int(len(aggregated_weights)/2)]
                aggregated_weights_2 = aggregated_weights[int(len(aggregated_weights)/2):]
                params_dict_1 = zip(net_1.state_dict().keys(), aggregated_weights_1)
                params_dict_2 = zip(net_2.state_dict().keys(), aggregated_weights_2)
                named_params_1 = [n for n, p in net_1.named_parameters()]
                named_params_2 = [n for n, p in net_2.named_parameters()]
                state_dict_1 = OrderedDict({k: torch.tensor(v) for k, v in params_dict_1 if k in named_params_1})
                state_dict_2 = OrderedDict({k: torch.tensor(v) for k, v in params_dict_2 if k in named_params_2})
                net_1.load_state_dict(state_dict_1, strict=False)
                net_2.load_state_dict(state_dict_2, strict=False)
                torch.save(
                    {
                        'round': rnd + self.round_offset,
                        'net_1_state_dict': net_1.state_dict(),
                        'net_2_state_dict': net_2.state_dict(),
                    }, self.model_file_name
                )
            else:
                params_dict = zip(net.state_dict().keys(), aggregated_weights)
                # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                named_params = [n for n, p in net.named_parameters()]
                state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if k in named_params})
                net.load_state_dict(state_dict, strict=False)
                torch.save(
                    {
                        'round': rnd + self.round_offset,
                        'net_state_dict': net.state_dict(),
                    }, self.model_file_name
                )

        if (rnd + self.round_offset) == self.total_rounds:
            i = 0
            while os.path.exists(self.loss_file_name[:-5]+"_exp_%s.json" %i):
                i += 1

            os.rename(self.loss_file_name, str(self.loss_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.all_loss_file_name, str(self.all_loss_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.plot_loss_file_name, str(self.plot_loss_file_name)[:-4]+"_exp_"+str(i)+".png")
            
        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ):# -> Tuple[Optional[float], Dict[str, Scalar]]:

        # Weigh accuracy of each client by number of examples used
        before_accuracies = [r.metrics["before_accuracy"] * r.num_examples for _, r in results]
        after_accuracies = [r.metrics["after_accuracy"] * r.num_examples for _, r in results]
        w_g_true_w_p_true = [r.metrics["w_g_true_w_p_true"] for _, r in results]
        w_g_true_w_p_false = [r.metrics["w_g_true_w_p_false"] for _, r in results]
        w_g_false_w_p_true = [r.metrics["w_g_false_w_p_true"] for _, r in results]
        w_g_false_w_p_false = [r.metrics["w_g_false_w_p_false"] for _, r in results]
        harmed = [r.metrics["harmed"] for _, r in results]#.count(True)
        examples = [r.num_examples for _, r in results]
        
        # Aggregate and print custom metric
        before_accuracy_aggregated = sum(before_accuracies) / sum(examples)
        after_accuracy_aggregated = sum(after_accuracies) / sum(examples)
        # misclassified_aggregated = sum(misclassified) / sum(examples)

        print(Fore.RED + f"{self.client_algorithm}: Round {rnd + self.round_offset} accuracy aggregated from client results: {after_accuracy_aggregated}" + Fore.WHITE)
        self.before_accuracies_dict[int(rnd + self.round_offset)] = before_accuracy_aggregated
        self.after_accuracies_dict[int(rnd + self.round_offset)] = after_accuracy_aggregated
        
        all_before_accuracies = [r.metrics["before_accuracy"] for _, r in results]
        self.all_before_accuracies_dict[int(rnd + self.round_offset)] = all_before_accuracies
        all_after_accuracies = [r.metrics["after_accuracy"] for _, r in results]
        self.all_after_accuracies_dict[int(rnd + self.round_offset)] = all_after_accuracies

        self.w_g_true_w_p_true_dict[int(rnd + self.round_offset)] = w_g_true_w_p_true
        self.w_g_true_w_p_false_dict[int(rnd + self.round_offset)] = w_g_true_w_p_false
        self.w_g_false_w_p_true_dict[int(rnd + self.round_offset)] = w_g_false_w_p_true
        self.w_g_false_w_p_false_dict[int(rnd + self.round_offset)] = w_g_false_w_p_false
        self.harmed_dict[int(rnd + self.round_offset)] = harmed
        self.sample_count_dict[int(rnd + self.round_offset)] = examples

        if self.dataset_name == 'stackoverflow_nwp':
            for c, r in results:
                if c.cid == '01165998':
                    self.highest_sampled_personalized_accuracy_dict[int(rnd + self.round_offset)] = r.metrics["after_accuracy"]
                elif c.cid == '03292394':
                    self.lowest_sampled_personalized_accuracy_dict[int(rnd + self.round_offset)] = r.metrics["after_accuracy"]
        elif self.dataset_name == 'emnist':
            for c, r in results:
                if c.cid == 'f0289_10':
                    self.highest_sampled_personalized_accuracy_dict[int(rnd + self.round_offset)] = r.metrics["after_accuracy"]
                elif c.cid == 'f1727_36':
                    self.lowest_sampled_personalized_accuracy_dict[int(rnd + self.round_offset)] = r.metrics["after_accuracy"]

        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.before_accu_file_name, 'w') as f:
                json.dump(self.before_accuracies_dict, f)
            with open(self.after_accu_file_name, 'w') as f:
                json.dump(self.after_accuracies_dict, f)

            with open(self.all_before_accu_file_name, 'w') as f:
                json.dump(self.all_before_accuracies_dict, f)
            with open(self.all_after_accu_file_name, 'w') as f:
                json.dump(self.all_after_accuracies_dict, f)

            with open(self.w_g_true_w_p_true_file_name, 'w') as f:
                json.dump(self.w_g_true_w_p_true_dict, f)
            with open(self.w_g_true_w_p_false_file_name, 'w') as f:
                json.dump(self.w_g_true_w_p_false_dict, f)
            with open(self.w_g_false_w_p_true_file_name, 'w') as f:
                json.dump(self.w_g_false_w_p_true_dict, f)
            with open(self.w_g_false_w_p_false_file_name, 'w') as f:
                json.dump(self.w_g_false_w_p_false_dict, f)

            with open(self.harmed_file_name, 'w') as f:
                json.dump(self.harmed_dict, f)
            with open(self.sample_count_file_name, 'w') as f:
                json.dump(self.sample_count_dict, f)

            with open(self.highest_sampled_personalized_accuracy_file_name, 'w') as f:
                json.dump(self.highest_sampled_personalized_accuracy_dict, f)
            with open(self.lowest_sampled_personalized_accuracy_file_name, 'w') as f:
                json.dump(self.lowest_sampled_personalized_accuracy_dict, f)

            plt.figure(2)
            plt.plot(self.before_accuracies_dict.keys(), self.before_accuracies_dict.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_before_accu_file_name)

            plt.figure(3)
            plt.plot(self.after_accuracies_dict.keys(), self.after_accuracies_dict.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_after_accu_file_name)

        if (rnd + self.round_offset) == self.total_rounds:
            i = 0
            while os.path.exists(self.after_accu_file_name[:-5]+"_exp_%s.json" %i):
                i += 1

            os.rename(self.before_accu_file_name, str(self.before_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.all_before_accu_file_name, str(self.all_before_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.after_accu_file_name, str(self.after_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.all_after_accu_file_name, str(self.all_after_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.plot_before_accu_file_name, str(self.plot_before_accu_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.plot_after_accu_file_name, str(self.plot_after_accu_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.model_file_name, str(self.model_file_name)[:-4]+"_exp_"+str(i)+".pth")
            os.rename(self.personalized_plot_accu_file_name, str(self.personalized_plot_accu_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.personalized_accu_file_name, str(self.personalized_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.personalized_all_accu_file_name, str(self.personalized_all_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            
            os.rename(self.w_g_true_w_p_true_file_name, str(self.w_g_true_w_p_true_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.w_g_true_w_p_false_file_name, str(self.w_g_true_w_p_false_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.w_g_false_w_p_true_file_name, str(self.w_g_false_w_p_true_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.w_g_false_w_p_false_file_name, str(self.w_g_false_w_p_false_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.harmed_file_name, str(self.harmed_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.sample_count_file_name, str(self.sample_count_file_name)[:-5]+"_exp_"+str(i)+".json")
            
            os.rename(self.highest_sampled_personalized_accuracy_file_name, str(self.highest_sampled_personalized_accuracy_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.lowest_sampled_personalized_accuracy_file_name, str(self.lowest_sampled_personalized_accuracy_file_name)[:-5]+"_exp_"+str(i)+".json")

        return super().aggregate_evaluate(rnd, results, failures)