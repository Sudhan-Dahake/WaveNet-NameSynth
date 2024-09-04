from Variables import *
from Vocabulary import *
from engine import *
from dependencies import *


words = open('names.txt', 'r').read().splitlines()

Vocab = Vocabulary(words)

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = Vocab(words[:n1])
Xte, Yte = Vocab(words[n1:n2])
Xdev, Ydev = Vocab(words[n2:])


torch.manual_seed(42)


model = Sequential(
    [
        Embeddings(Vocab.vocab_size, n_embed),
        FlattenConsecutive(2), Linear(2 * n_embed, n_hidden),          BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(2 * n_hidden, n_hidden),         BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(2 * n_hidden, Vocab.vocab_size),
    ]
)

with torch.no_grad():
    for layer in model.layers:
        if isinstance(layer, Linear):
            layer.weights *= 5/3

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True



# Training the model
for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')




for layer in model.layers:
    layer.training = False




def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


split_loss('train')
split_loss('val')


# Function to save the model's state, including buffers
def save_model(model, filepath):
    model_state = {}
    for i, layer in enumerate(model.layers):
        layer_state = {
            'parameters': [p.clone() for p in layer.parameters()]
        }
        # Save buffers such as running_mean, running_var, etc.
        if hasattr(layer, 'running_mean'):
            layer_state['running_mean'] = layer.running_mean.clone()
        if hasattr(layer, 'running_var'):
            layer_state['running_var'] = layer.running_var.clone()
        if hasattr(layer, 'training'):
            layer_state['training'] = layer.training
        model_state[f'layer_{i}'] = layer_state

    torch.save(model_state, filepath)


# Save the model state after training
save_model(model, 'model_state.pth')
