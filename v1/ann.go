package main

import (
	"fmt"
	"math"
)

type ANN struct {
	layers [][]Neuron
}

func (ann *ANN) init(numNeurons []int) {
	ann.layers = make([][]Neuron, len(numNeurons)-1)

	for i, neuronsInLayer := range numNeurons {
		if i == 0 {
			// Ignore input layer, since no transformation happens here
			continue
		}

		layer := make([]Neuron, neuronsInLayer)

		for j := 0; j < neuronsInLayer; j++ {

			neuron := Neuron{}
			neuron.init(numNeurons[i-1])

			layer[j] = neuron
		}

		ann.layers[i-1] = layer
	}
}

func (ann *ANN) Predict(values []float64) ([]float64, [][]float64) {
	previousLayerOutputs := values

	neuronActivations := make([][]float64, len(ann.layers))

	for i, layer := range ann.layers {

		neuronActivations[i] = make([]float64, len(layer))

		layerOutputs := make([]float64, len(layer))

		for j, neuron := range layer {
			neuronOutput := neuron.Forward(previousLayerOutputs)

			layerOutputs[j] = neuronOutput
			neuronActivations[i][j] = neuronOutput
		}

		previousLayerOutputs = layerOutputs
	}

	return previousLayerOutputs, neuronActivations
}

func (ann *ANN) Train(dataPoints []DataPoint, epochs int) {

	for epoch := 1; epoch <= epochs; epoch++ {

		epochLoss := 0.0

		for _, dataPoint := range dataPoints {

			var predictions, neuronActivations = ann.Predict(dataPoint.x)

			// fmt.Println(predictions)
			// fmt.Println(neuronActivations)

			if len(predictions) != len(dataPoint.y) {
				panic("Received inproper amount of output values!")
			}

			incomingDeltas := make([]float64, len(predictions))
			for n := 0; n < len(predictions); n++ {
				incomingDelta := predictions[n] - dataPoint.y[n]
				incomingDeltas[n] = incomingDelta

				epochLoss += 0.5 * math.Pow((predictions[n]-dataPoint.y[n]), 2)
			}

			for i := len(ann.layers) - 1; i >= 0; i-- {

				var previousActivations []float64
				var layerDeltas []float64

				if i > 0 {
					previousActivations = neuronActivations[i-1]
					layerDeltas = make([]float64, len(ann.layers[i-1]))
				} else {
					previousActivations = dataPoint.x
					layerDeltas = make([]float64, len(dataPoint.x))
				}

				for _, neuron := range ann.layers[i] {
					previousDeltas := neuron.Backwards(incomingDeltas, previousActivations)

					for k, previousDelta := range previousDeltas {
						layerDeltas[k] += previousDelta
					}

				}

				incomingDeltas = layerDeltas
			}
		}

		if epoch%2 == 0 {
			fmt.Printf("Loss after %d epoch: %v\n\n", epoch, epochLoss)
		}
	}
}
