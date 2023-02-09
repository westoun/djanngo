package network

import (
	. "djanngo/v3/layers"
)

type Network struct {
	Layers []Layer
}

func (network *Network) Predict(batch [][]float64) [][]float64 {
	output := batch
	for _, layer := range network.Layers {
		output = layer.Forward(output)
	}

	return output
}

func (network *Network) UpdateGradients(batchLoss [][]float64) {

	layerCount := len(network.Layers)
	lastLayer := network.Layers[layerCount-1]

	lastLayer.Backward(batchLoss)
}

func (network Network) String() string {
	stringified := ""

	for _, layer := range network.Layers {
		stringified += "\n" + layer.String() + "\n"
	}

	return stringified
}
