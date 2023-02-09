# DjANNGo

Similar to the origin story of [Apache Lucene](https://en.wikipedia.org/wiki/Apache_Lucene),
I created DjANNGo first and foremost as a learning project for
myself.
The main objectives were to a) get some hands-on experience in golang,
and b) deepen my understanding of the components and pitfalls of developing
deep neural networks.
As such, it serves as a documented playground, rather than a robust framework
ready for application.

## Versions and Changes

**Note:** Instead of refactoring the whole codebase on every learning, I went
with a tabula-rasa approach and start a new version whenever fundamental
assumptions or implementation ideas change. Still, I decided to keep the
old versions of the framework as a form of "experience log".

The following paragraphs comprise the core ideas and changes between
versions.

### Version 3

The layer is the atomic unit of the network, with linear layers and
activation layers being separated for the first time.
Each layer receives a reference to its immediate predecessor, to which
it independently passes on its backward gradient. The network merely
acts as a wrapper around the layers and an entry point for updating.

Updatable parameters, such as the weights and biases within a linear
layer, are stored as instances of a distinct interface: the updatable.
Aside from its values, each updatable stores its gradient, computed
and accessed by the surrounding layer. While the gradients are _set_
by the layer, they are _applied_ by an external optimizer.

### Version 2

Everything happens on the level of the network. The structure of
layers is passed as a list of integers, weights and biases as stored
as multidimensional arrays. Gradients are computed by the network,
modified by a learning rate scheduler and optimizer, and the passed
back to the network for updating.

### Version 1

The neuron is the atomic unit of the network. The activation function
is part of the neuron. Backwards passing of gradients is facilitated
by the network, while the updates of weights and biases happens within
the neuron, as well as the computation of gradients to pass on. Each
neuron only knows of itself.

## Installation

DjANNGo does not use any external dependencies.

As long as you have [go installed](https://go.dev/learn/), you
should be "good to go" (🥁).

## Usage

The usage of DjANNGo varies by version.
A working example for each version can be found within the corresponding
[main.go](./v3/main.go) file.
In general, it is adviced to check the latest version first, since it
encompasses the lessons learned from previous approaches.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
