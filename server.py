from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics

aggregated_accuracies = []

def save_accuracies_to_file(filename: str, accuracies: List[float]) -> None:
    with open(filename, "w") as file:
        for i, accuracy in enumerate(accuracies):
            file.write(f"Round {i}: {accuracy}\n")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    aggregated_accuracy = sum(accuracies) / sum(examples)
    aggregated_accuracies.append(aggregated_accuracy)
    return {"accuracy": aggregated_accuracy}

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=15),
    strategy=strategy,
)

save_accuracies_to_file("accuracies_log.txt", aggregated_accuracies)

print("Aggregated accuracies:", aggregated_accuracies)
