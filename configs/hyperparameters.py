from dataloaders.synthetic.generate_synthetic_data import NUM_USER 
RANDOM_SEED = 1

stackoverflow_nwp_local = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/Local/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_fedavg = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/FedAvg/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_fedavgft = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/FedAvgFT/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_knnper = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'knn_neighbor_count': 5,
    'lambda_': 0.5,
    'checkpoint_dir': './results/stackoverflow_nwp/knnPer/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_partialfed = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/PartialFed/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_apfl = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'alpha': 0.25,
    'checkpoint_dir': './results/stackoverflow_nwp/APFL/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_ditto = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'lambda_': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/Ditto/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_fedrep = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/FedRep/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_lgfedavg = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/LGFedAvg/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_hypcluster = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/HypCluster/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_flow = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'checkpoint_dir': './results/stackoverflow_nwp/Flow/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_flowv2 = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/stackoverflow_nwp/FlowV2/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_flowv3 = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/stackoverflow_nwp/FlowV3/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_flowv4 = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/stackoverflow_nwp/FlowV4_seed_9/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_flowv5 = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/stackoverflow_nwp/FlowV5/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_nwp_flowv6 = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.1,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/stackoverflow_nwp/FlowV6/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

emnist_local = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 10,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.05,
    'checkpoint_dir': './results/emnist/Local/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_fedavg = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'checkpoint_dir': './results/emnist/FedAvg/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_fedavgft = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'checkpoint_dir': './results/emnist/FedAvgFT/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_knnper = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'knn_neighbor_count': 10,
    'lambda_': 0.4,
    'checkpoint_dir': './results/emnist/knnPer/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_partialfed = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'checkpoint_dir': './results/emnist/PartialFed/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_apfl = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'alpha': 0.25,
    'checkpoint_dir': './results/emnist/APFL/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_ditto = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'lambda_': 0.1,
    'checkpoint_dir': './results/emnist/Ditto/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_fedrep = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'checkpoint_dir': './results/emnist/FedRep/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_lgfedavg = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'checkpoint_dir': './results/emnist/LGFedAvg/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_hypcluster = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'checkpoint_dir': './results/emnist/HypCluster/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_flowv2 = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'lambda_': 0.01,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/emnist/FlowV2/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_flowv3 = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'lambda_': 0.01,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/emnist/FlowV3/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_flowv4 = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'lambda_': 0.01,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/emnist/FlowV4/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_flowv5 = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'lambda_': 0.01,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/emnist/FlowV5/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

emnist_flowv6 = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 3,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.01,
    'lambda_': 0.01,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/emnist/FlowV6/',
    'dataset_dir': './dataloaders/emnist/',
    'checkpoint_interval': 10,
}

cifar10_local = {
    'num_clients': 10,
    'total_clients': 100,
    'rounds': 4000,
    'epochs': 20,
    'batch_size': 20,
    'lr': 0.1,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/Local/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_fedavg = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FedAvg/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_fedavgft = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FedAvgFT/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_knnper = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'knn_neighbor_count': 5,
    'lambda_': 0.5,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/knnPer/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_partialfed = {
    'num_clients': 10,
    'total_clients': 100,
    'rounds': 4000,
    'epochs': 5,
    'batch_size': 20,
    'lr': 0.11,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/PartialFed/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_apfl = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'alpha': 0.25,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/APFL/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_ditto = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.01,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/Ditto/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_fedrep = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FedRep/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_lgfedavg = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/LGFedAvg/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_hypcluster = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/HypCluster/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_flowv2 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FlowV2/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_flowv3 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FlowV3/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_flowv4 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FlowV4/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_flowv5 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FlowV5/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar10_flowv6 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar10_0_6/FlowV6/',
    'dataset_dir': './dataloaders/cifar10/',
    'checkpoint_interval': 10,
}

cifar100_local = {
    'num_clients': 10,
    'total_clients': 100,
    'rounds': 4000,
    'epochs': 20,
    'batch_size': 20,
    'lr': 0.1,
    'dirichlet_parameter': 0.1,
    'checkpoint_dir': './results/cifar100_0_1/Local/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_fedavg = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/FedAvg/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_fedavgft = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/FedAvgFT/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_knnper = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'knn_neighbor_count': 5,
    'lambda_': 0.4,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/knnPer/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_partialfed = {
    'num_clients': 10,
    'total_clients': 100,
    'rounds': 4000,
    'epochs': 5,
    'batch_size': 20,
    'lr': 0.11,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/PartialFed/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_apfl = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'alpha': 0.25,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/APFL/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_ditto = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.01,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/Ditto/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_fedrep = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/FedRep/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_lgfedavg = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/LGFedAvg/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_hypcluster = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/HypCluster/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_flowv2 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/FlowV2/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_flowv3 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/FlowV3/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_flowv4 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.1,
    'checkpoint_dir': './results/cifar100_0_1/FlowV4/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_flowv5 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/FlowV5/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

cifar100_flowv6 = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 3,
    'batch_size': 20,
    'lr': 0.05,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'dirichlet_parameter': 0.6,
    'checkpoint_dir': './results/cifar100_0_6/FlowV6/',
    'dataset_dir': './dataloaders/cifar100/',
    'checkpoint_interval': 10,
}

shakespeare_local = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'checkpoint_dir': './results/shakespeare/Local/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_fedavg = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'checkpoint_dir': './results/shakespeare/FedAvg/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_fedavgft = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'checkpoint_dir': './results/shakespeare/FedAvgFT/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_knnper = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'knn_neighbor_count': 3,
    'lambda_': 0.5,
    'checkpoint_dir': './results/shakespeare/knnPer/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_partialfed = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'checkpoint_dir': './results/shakespeare/PartialFed/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_apfl = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'alpha': 0.25,
    'checkpoint_dir': './results/shakespeare/APFL/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_ditto = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'lambda_': 0.1,
    'checkpoint_dir': './results/shakespeare/Ditto/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_fedrep = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'lambda_': 0.1,
    'checkpoint_dir': './results/shakespeare/FedRep/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_lgfedavg = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'lambda_': 0.1,
    'checkpoint_dir': './results/shakespeare/LGFedAvg/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_hypcluster = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'lambda_': 0.1,
    'checkpoint_dir': './results/shakespeare/HypCluster/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_flow = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.1,
    'checkpoint_dir': './results/shakespeare/Flow/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_flowv2 = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.11,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/shakespeare/FlowV2/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_flowv3 = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.11,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/shakespeare/FlowV3/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_flowv4 = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.11,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/shakespeare/FlowV4/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_flowv5 = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.11,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/shakespeare/FlowV5/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

shakespeare_flowv6 = {
    'num_clients': 10,
    'total_clients': 715,
    'rounds': 1500,
    'epochs': 5,
    'embedding_size': 8, 
    'vocab_size': 90, 
    'hidden_size': 256,
    'batch_size': 4,
    'lr': 0.11,
    'lambda_': 0.001,
    'epsilon': 1e-6,
    'checkpoint_dir': './results/shakespeare/FlowV6/',
    'dataset_dir': './dataloaders/shakespeare/',
    'checkpoint_interval': 10,
}

#########################

synthetic_local = {
    'num_clients': 10,
    'total_clients': NUM_USER,
    'rounds': 1000,
    'epochs': 10,
    'batch_size': 10,
    'input_dim': 60,
    'hidden_dim': 20,
    'output_dim': 10,
    'lr': 0.01,
    'checkpoint_dir': './results/synthetic/Local/',
    'dataset_dir': './dataloaders/synthetic/',
    'checkpoint_interval': 10,
}

synthetic_fedavg = {
    'num_clients': 10,
    'total_clients': NUM_USER,
    'rounds': 1000,
    'epochs': 1,
    'batch_size': 10,
    'input_dim': 60,
    'hidden_dim': 20,
    'output_dim': 10,
    'lr': 0.01,
    'checkpoint_dir': './results/synthetic/FedAvg/',
    'dataset_dir': './dataloaders/synthetic/',
    'checkpoint_interval': 10,
}

synthetic_fedavgft = {
    'num_clients': 10,
    'total_clients': NUM_USER,
    'rounds': 1000,
    'epochs': 1,
    'batch_size': 10,
    'input_dim': 60,
    'hidden_dim': 20,
    'output_dim': 10,
    'lr': 0.01,
    'checkpoint_dir': './results/synthetic/FedAvgFT/',
    'dataset_dir': './dataloaders/synthetic/',
    'checkpoint_interval': 10,
}

synthetic_knnper = {
    'num_clients': 10,
    'total_clients': NUM_USER,
    'rounds': 1000,
    'epochs': 1,
    'batch_size': 10,
    'input_dim': 60,
    'hidden_dim': 20,
    'output_dim': 10,
    'lr': 0.01,
    'checkpoint_dir': './results/synthetic/knnPer/',
    'dataset_dir': './dataloaders/synthetic/',
    'checkpoint_interval': 10,
}

stackoverflow_lr_local = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 1500,
    'epochs': 5,
    'batch_size': 32,
    'word_vocab_size': 10000,
    'tag_vocab_size': 500,
    'lr': 0.01,
    'checkpoint_dir': './results/stackoverflow_lr/Local/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10000,
}

stackoverflow_lr_fedavg = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 1500,
    'epochs': 5,
    'batch_size': 32,
    'word_vocab_size': 10000,
    'tag_vocab_size': 500,
    'lr': 0.01,
    'checkpoint_dir': './results/stackoverflow_lr/FedAvg/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_lr_fedavgft = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 1500,
    'epochs': 5,
    'batch_size': 32,
    'word_vocab_size': 10000,
    'tag_vocab_size': 500,
    'lr': 0.01,
    'checkpoint_dir': './results/stackoverflow_lr/FedAvgFT/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}

stackoverflow_lr_knnper = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 1500,
    'epochs': 5,
    'batch_size': 32,
    'word_vocab_size': 10000,
    'tag_vocab_size': 500,
    'lr': 0.01,
    'checkpoint_dir': './results/stackoverflow_lr/knnPer/',
    'dataset_dir': './dataloaders/stackoverflow/',
    'checkpoint_interval': 10,
}


