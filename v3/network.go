package main

import . "djanngo/v3/layers"

type Network struct {
	layers []Layer
}

func (network *Network) Predict(batch [][]float64) [][]float64 {
	output := batch
	for _, layer := range network.layers {
		output = layer.Forward(output)
	}

	return output
}

func (network *Network) UpdateGradients(batchLoss [][]float64) {

	layerCount := len(network.layers)
	lastLayer := network.layers[layerCount-1]

	lastLayer.Backward(batchLoss)
}
