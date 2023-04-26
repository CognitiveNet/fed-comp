import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision.datasets import CIFAR10
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


import warnings

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = []
    testset = []

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / 10)
    n_test = int(num_examples["testset"] / 10)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)


def train(model, trainloader, valloader, epochs, device: str = "cuda:0"):
    """Train the network on the training set."""
    print("Starting training...")
    print("Epochs totais:", epochs)
    model.train()
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )


    for e in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print("epochs", e)
        print("val_loss", loss)


    model.to("cpu")  # move model back to CPU

    train_loss, train_acc, _,_ = test(model, trainloader)
    val_loss, val_acc,_,_ = test(model, valloader)




    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(model, dataloader, steps: int = None, device: str = "cuda:0"):
    print("Starting testing...")
    model.eval()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)


    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, labels in dataloader:
            #print("data", data.shape)
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels).to(device)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.to("cpu").numpy())
            all_labels.extend(labels.to("cpu").numpy())



    accuracy = accuracy_score(all_labels, all_preds).item()
    loss = loss.cpu().numpy().min().item()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("accuracy", accuracy)
    print("loss", loss)
    print("f1", f1)
    print("conf_matrix", conf_matrix)

    return loss, accuracy, f1, conf_matrix


def replace_classifying_layer(efficientnet_model, num_classes: int = 7):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)


def load_model(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None):

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    if classes is not None:
        model.classifier = torch.nn.Linear(model.classifier[0].in_features, classes)
    return model


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]