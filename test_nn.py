import nn
import activations as a
import transforms as t

mod = nn.nn()
mod.model(
    t.Linear(784,128),
    a.LeakyReLU(),
    t.Linear(128,64),
    a.LeakyReLU(),
    t.Linear(64,10),
    a.softmax()
)

print(mod)

def test_dimensions():
    assert mod.fns[0].weights.shape[0] == 128 and mod.fns[0].weights.shape[1] == 784
    assert mod.fns[2].weights.shape[0] == 64 and mod.fns[2].weights.shape[1] == 128
    assert mod.fns[4].biases.shape[0] == 10


