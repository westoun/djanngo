package main

import (
	"math/rand"
)

/*
	Assumption: only sigmoid activation (for now)
*/

type Neuron struct {
	weights []float64
	bias    float64
}

func (n *Neuron) init(numWeights int) {

	n.weights = make([]float64, numWeights)

	for i := 0; i < numWeights; i++ {
		n.weights[i] = rand.Float64()
	}

	n.bias = rand.Float64()
}

func (n *Neuron) Forward(values []float64) float64 {
	if len(values) != len(n.weights) {
		// TODO: Define better error handling and naming.
		panic("Received inproper amount of input values!")
	}

	activationPotential := 0.0
	for i := 0; i < len(values); i++ {
		activationPotential += n.weights[i] * values[i]
	}
	activationPotential += n.bias

	activation := Sigmoid(activationPotential)
	return activation
}

func (n *Neuron) Backwards(incomingDeltas []float64, previousActivations []float64) []float64 {

	learningRate := 0.1

	prediction := n.Forward(previousActivations)

	outgoingDeltas := make([]float64, len(n.weights))

	for _, incomingDelta := range incomingDeltas {

		for i := 0; i < len(n.weights); i++ {
			outgoingDeltas[i] = n.weights[i] * Sigmoid(prediction) * (1 - Sigmoid(prediction)) * incomingDelta

			// TA: learning rate?
			n.weights[i] -= learningRate * previousActivations[i] * Sigmoid(prediction) * (1 - Sigmoid(prediction)) * incomingDelta
		}

		n.bias -= learningRate * Sigmoid(prediction) * (1 - Sigmoid(prediction)) * incomingDelta

	}

	return outgoingDeltas
}
