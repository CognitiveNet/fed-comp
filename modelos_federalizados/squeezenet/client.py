import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")
# Definir transformações


class Client(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        validation_split: int = 0.1,
        partition: int = 0

    ):
        self.device = "cuda:0"
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.partition = partition

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = utils.load_model(classes=7)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Criar objetos ImageFolder para cada pasta de treino e teste
        train_dataset = ImageFolder('D:/dataset/dataset_split' + str(self.partition)+'/train', transform=transform)


        # Dividir o conjunto de treino em treino e validação
        train_size = len(train_dataset)
        val_size = int(0.1 * train_size)
        train_size = train_size - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Criar dataloaders de treino, teste e validação
        batch_size = 8
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        # Update local model parameters
        model = self.set_parameters(parameters)


        # Get hyperparameters for this round
        batch_size = 8
        epochs = 10

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )


        results = utils.train(model, train_dataloader, val_dataloader, epochs, self.device)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        batch_size = 16
        test_dataset = ImageFolder('D:/dataset/dataset_split' + str(self.partition)+'/test', transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Get config values
        steps: int = config["val_steps"]


        loss, accuracy,_,_ = utils.test(model, test_dataloader, steps, self.device)

        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(device: str = "cuda:0"):
    """Weak tests to check whether all client methods are working as
    expected."""

    model = utils.load_model(classes=7)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    client = Client(trainset, testset, device)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 8, "local_epochs": 10},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})

    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.dry:
        client_dry_run(device)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        trainset, testset = utils.load_partition(args.partition)

        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        # Start Flower client
        client = Client(trainset, testset, device, partition=args.partition)

        fl.client.start_numpy_client(server_address="127.0.0.1:8082", client=client)
        fl.client.start_numpy_client(server_address="127.0.0.1:8082", client=client)


if __name__ == "__main__":
    main()