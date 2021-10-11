import torch

from robustbench import benchmark
from robustbench.utils import load_model

threat_model = "corruptions"
dataset = "cifar10"

model_name = 'Diffenderfer2021Winning_Binary_CARD_Deck'
model = load_model(model_name=model_name, dataset=dataset, threat_model=threat_model)
device = torch.device("cuda:0")

clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                  threat_model=threat_model, device=device, to_disk=True)

