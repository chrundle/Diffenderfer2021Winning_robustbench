from robustbench.data import load_cifar10, load_cifar10c, load_cifar100, load_cifar100c
from robustbench.utils import load_model, clean_accuracy

dataset = 'cifar10'
threat_model = 'corruptions'

if dataset == 'cifar10':
  x_test, y_test = load_cifar10()
  #x_test, y_test = load_cifar10c(n_examples=100)
elif dataset == 'cifar100':
  x_test, y_test = load_cifar100()
  #x_test, y_test = load_cifar100c(n_examples=100)


#model_name = 'Diffenderfer2021Winning_LRR'
#model_name = 'Diffenderfer2021Winning_Binary'
model_name = 'Diffenderfer2021Winning_LRR_CARD_Deck'
#model_name = 'Diffenderfer2021Winning_Binary_CARD_Deck'
model = load_model(model_name=model_name, dataset=dataset, threat_model='corruptions')
print(f'Testing model {model_name} on {dataset}...')
acc = clean_accuracy(model.cuda(), x_test, y_test, device='cuda:0')
print(f'Model: {model_name}, {dataset} accuracy: {acc:.1%}')
