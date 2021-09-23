from robustbench.data import load_cifar10

x_test, y_test = load_cifar10()
#x_test, y_test = load_cifar10(n_examples=50)

from robustbench.utils import load_model, clean_accuracy

model_name = 'Diffenderfer2021Winning_LRR_CARD_Deck'
model = load_model(model_name=model_name, dataset='cifar10', threat_model='corruptions')
print(f'Testing model {model_name} on CIFAR-10...')
acc = clean_accuracy(model.cuda(), x_test, y_test, device='cuda:0')
print(f'Model: {model_name}, CIFAR-10 accuracy: {acc:.1%}')
