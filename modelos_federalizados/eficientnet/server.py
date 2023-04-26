from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import flwr as fl
import torch
from torchvision.datasets import ImageFolder
import csv

import utils

import warnings
import numpy as np

warnings.filterwarnings("ignore")


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 8,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Criar objetos ImageFolder para cada pasta de treino e teste
    train_dataset = ImageFolder('D:/dataset/dataset_split' + str(0) + '/train', transform=transform)

    # Dividir o conjunto de treino em treino e validação
    train_size = len(train_dataset)
    val_size = int(0.1 * train_size)
    train_size = train_size - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Criar dataloaders de treino, teste e validação
    batch_size = 8
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy, f1, conf_matrix = utils.test(model, val_dataloader)


        # caso não exista o arquivo, cria um, e adiciona o f1 score na ultima linha
        with open('f1.csv', mode='a', newline='') as arquivo:
            escritor_csv = csv.writer(arquivo)
            escritor_csv.writerow([f1])

        # caso não exista o arquivo, cria uma matriz de confusão
        with open('conf_matrix.csv', mode='w', newline='') as arquivo:
            escritor_csv = csv.writer(arquivo)
            escritor_csv.writerow([conf_matrix])

        # caso não exista o arquivo, cria um, e adiciona o loss score na ultima linha
        with open('loss.csv', mode='a', newline='') as arquivo:
            escritor_csv = csv.writer(arquivo)
            escritor_csv.writerow([loss])

        # caso não exista o arquivo, cria um, e adiciona a accuracy score na ultima linha
        with open('accuracy.csv', mode='a', newline='') as arquivo:
            escritor_csv = csv.writer(arquivo)
            escritor_csv.writerow([accuracy])


        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )

    args = parser.parse_args()

    model = utils.load_model(classes=7)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8082",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()