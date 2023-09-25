
# Flow: Per-Instance Personalized Federated Learning

Flow addresses the challenge of statistical heterogeneity in federated learning through creating a dynamic personalized model for each input instance through a routing mechanism. Our contributions are threefold:

( a ) We propose a per-instance and per-client personalization approach Flow that creates personalized models via dynamic routing, which improves both the performance of the personalized model and the generalizability of the global model.

( b ) We derive convergence analysis for both global and personalized models, showing how the routing policy influences convergence rates based on the across- and within- client heterogeneity.

( c ) We empirically evaluate the superiority of Flow in both generalization and personalized accuracy on various vision and language tasks in cross-device FL settings.
  

## Download Datasets

1. Stackoverflow NWP:

```./dataloaders``` contains a modified [data loader](https://github.com/google-research/federated/blob/491c8cb90533ed075d286e1edffebf8fd3ac1dac/utils/datasets/stackoverflow_word_prediction.py) for Stackoverflow dataset, which has been hosted on [TensorFlow Federated library](tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data).

```./dataloaders/stackoverflow``` directory contains list of all the available clients which are used for training, validation, and testing. This dataloader is for Stackoverflow Next Word Prediction Task for a non-convex RNN model.

 
2. Stackoverflow LR:

```./dataloaders``` contains a modified [data loader](https://github.com/google-research/federated/blob/491c8cb90533ed075d286e1edffebf8fd3ac1dac/utils/datasets/stackoverflow_tag_prediction.py) for Stackoverflow LR dataset.

```./dataloaders/stackoverflow``` directory contains list of all the available clients which are used for training, validation, and testing. This dataloader is for Stackoverflow Tag Prediction Task for a convex logistic regression baseline.

3. EMNIST:

```./datalaoders``` contains a modified [data loader](https://github.com/google-research/federated/blob/491c8cb90533ed075d286e1edffebf8fd3ac1dac/utils/datasets/emnist_dataset.py) for EMNIST dataset, which has been hosted on [TensorFlow Federated library](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data).

```./dataloaders/emnist``` directory contains list of all the available clients which are used for training, validation, and testing. This dataloader is for EMNIST Image Classification Task for a non-convex CNN baseline.

4. Shakespeare:

```./dataloaders``` contains a modified [data loader](https://github.com/google-research/federated/blob/491c8cb90533ed075d286e1edffebf8fd3ac1dac/utils/datasets/shakespeare_dataset.py) for Shakespeare dataset, which has been hosted on [TensorFlow Federated library](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data).

```./dataloaders/shakespeare``` directory contains list of all the available clients which are used for training, validation, and testing. This dataloader is for Shakespeare Next Character Prediction Task for a non-convex RNN baseline.
  

5. CIFAR-100 (and CIFAR10):

```./dataloaders``` contains a modified [data loader](https://github.com/google-research/federated/blob/491c8cb90533ed075d286e1edffebf8fd3ac1dac/utils/datasets/cifar100_dataset.py) for CIFARA-100 dataset, which has been hosted on [TensorFlow Federated library](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data).

```./dataloaders/cifar100``` directory contains list of all the available clients which are used for training, validation, and testing. This dataloader is for CIFAR-100 Image Classification Task for a non-convex CNN baseline.

  
## Models

Each dataset/task would have its corresponding model in ```./models```. Each model file contains a model definition and its forward pass method. The models from ```./models``` and dataloaders from ```./dataloaders``` are referred to from client trainer files and server aggregator files, which are inside ```./trainers``` directory.

## Run Baselines and Flow

Each baseline among

- FedAvg - https://arxiv.org/pdf/2012.06706.pdf

- FedAvgFT

- APFL - https://arxiv.org/pdf/2003.13461.pdf

- Ditto - https://arxiv.org/pdf/2012.04221.pdf

- FedRep - https://arxiv.org/pdf/2102.07078.pdf

- HypCluster - https://arxiv.org/pdf/2002.10619.pdf

- knnPer - https://arxiv.org/pdf/2111.09360.pdf

- LGFedAvg - https://arxiv.org/pdf/2001.01523.pdf

have their separate client-side trainers in ```./trainers```.

Our approach, Flow is implemented similar to its baselines in ```./trainers```.


**The entry point of each federated training standalone simulation execution is the client file, which is inside ```./trainers```.**

  

First, one would need to create checkpoint and results directory in ```./results``` directory, with a subdirectory named after the baseline name (e.g., ```./results/<dataset_name>/<method_name>``` to store results of FedAvg baselines which you would run from ```./trainers/client_fedavg.py```). This would store the clients/server intermediate results.

  

From the root directory, one needs to type the following command to start a simulation:

  

```python3 ./trainers/client_\<baseline name\>.py --dataset \<dataset name\> ```

  

\<dataset  name\> can be one of these "stackoverflow_nwp", "stackoverflow_lr", "emnist", "shakespeare", "emnist", "cifar100", "synthetic".

  

One can change the hyperparameters for any trainer and dataset combo in ```./configs/hyperpatrameters.py``` file.

  

## Single Client Tests

Flow is built on [Flower](https://flower.dev/docs/) framework, for Pytorch.

Because of the way Flower prints logs on the terminal and handles errors/exceptions, one might have trouble debugging some parts of the code.
Easiest solution would be to run the same functions for a single client.

Hence in each ```client_<baseline trainer name>.py``` file, one can comment out the

```fl.simulation.start_simulation``` call and uncomment the single client execution code below it. There are codechunks for all the datasets in ```./trainers/client_fedavg.py```, one can have similar single client tests for other baseline trainers too.