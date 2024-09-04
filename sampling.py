from dependencies import *
from Variables import *
from engine import *
from Vocabulary import *


words = open('names.txt', 'r').read().splitlines()

Vocab = Vocabulary(words)


# Recreate the model with the same architecture
model = Sequential(
    [
        Embeddings(Vocab.vocab_size, n_embed),
        FlattenConsecutive(2), Linear(2 * n_embed, n_hidden),          BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(2 * n_hidden, n_hidden),         BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(2 * n_hidden, n_hidden),         BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(2 * n_hidden, n_hidden),         BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(2 * n_hidden, Vocab.vocab_size),
    ]
)



# Function to load the model's state, including buffers
def load_model(model, filepath):
    model_state = torch.load(filepath)
    for i, layer in enumerate(model.layers):
        layer_state = model_state[f'layer_{i}']
        params = layer_state['parameters']
        for param, saved_param in zip(layer.parameters(), params):
            param.data.copy_(saved_param)
        # Load buffers such as running_mean, running_var, etc.
        if (('running_mean' in layer_state) and (isinstance(layer, BatchNorm1d))):
            if layer.running_mean.shape != layer_state['running_mean'].shape:
                reshaped_running_mean = layer_state['running_mean'].view_as(
                    layer.running_mean)
                layer.running_mean.data.copy_(reshaped_running_mean)
            else:
                layer.running_mean.data.copy_(layer_state['running_mean'])
        if (('running_var' in layer_state) and (isinstance(layer, BatchNorm1d))):
            if layer.running_var.shape != layer_state['running_var'].shape:
                reshaped_running_var = layer_state['running_var'].view_as(
                    layer.running_var)
                layer.running_var.data.copy_(reshaped_running_var)
            else:
                layer.running_var.data.copy_(layer_state['running_var'])
        if (('training' in layer_state) and (isinstance(layer, BatchNorm1d))):
            layer.training = layer_state['training']


# Load the saved model state
load_model(model, 'model_state.pth')


# sampling from the dataset
g = torch.Generator().manual_seed(2147483647 + 10)

for i in range(20):
    out = []
    context = [0] * block_size

    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        out.append(ix)
        if ix == 0:
            break
        context = context[1:] + [ix]

    print(''.join(Vocab.itos[i] for i in out))
