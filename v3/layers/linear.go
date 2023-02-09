package layers

import (
	. "djanngo/v3/updatables"
	"fmt"
	"math/rand"
)

type Linear struct {
	parents   []Layer
	weights   UpdatableMatrix
	biases    UpdatableVector
	lastBatch [][]float64
}

func (layer *Linear) Init(inSize int, layerSize int, parents []Layer) {
	layer.parents = parents

	layer.weights = UpdatableMatrix{}
	layer.weights.Values = make([][]float64, layerSize)
	// layer.weights.Gradients = make([][][]float64, 0)

	for l := 0; l < layerSize; l++ {
		layer.weights.Values[l] = make([]float64, inSize)
		for w := 0; w < inSize; w++ {
			layer.weights.Values[l][w] = rand.Float64() - 0.5
		}
	}

	layer.biases = UpdatableVector{}
	layer.biases.Values = make([]float64, layerSize)
	for l := 0; l < layerSize; l++ {
		layer.biases.Values[l] = rand.Float64() - 0.5
	}
}

func (layer *Linear) Forward(batch [][]float64) [][]float64 {
	layer.lastBatch = batch

	output := make([][]float64, len(batch))

	for i, values := range batch {
		output[i] = make([]float64, len(layer.weights.Values))

		for n := 0; n < len(layer.weights.Values); n++ {

			for w := 0; w < len(layer.weights.Values[n]); w++ {
				output[i][n] += layer.weights.Values[n][w] * values[w]
			}
			output[i][n] += layer.biases.Values[n]
		}

	}

	return output
}

func (layer *Linear) Backward(incomingGradients [][]float64) {

	// weight gradients
	weightGradients := make([][]float64, len(layer.weights.Values))
	for n := 0; n < len(layer.weights.Values); n++ {
		weightGradients[n] = make([]float64, len(layer.weights.Values[n]))
	}

	for i := 0; i < len(incomingGradients); i++ {

		for n := 0; n < len(layer.weights.Values); n++ {

			for w := 0; w < len(layer.weights.Values[n]); w++ {

				weightGradients[n][w] += incomingGradients[i][n] * layer.lastBatch[i][w]

			}

		}

	}
	layer.weights.Gradients = weightGradients

	// bias gradients
	biasGradients := make([]float64, len(layer.biases.Values))
	for i := 0; i < len(incomingGradients); i++ {
		for j := 0; j < len(incomingGradients[i]); j++ {
			biasGradients[j] += incomingGradients[i][j]
		}
	}
	layer.biases.Gradients = biasGradients

	// backpropagate
	backwardGradients := make([][]float64, len(incomingGradients))
	for i := 0; i < len(layer.lastBatch); i++ {
		backwardGradients[i] = make([]float64, len(layer.lastBatch[i]))

		for j := 0; j < len(layer.lastBatch[i]); j++ {

			for n := 0; n < len(layer.weights.Values); n++ {

				backwardGradients[i][j] += incomingGradients[i][n] *
					layer.weights.Values[n][j]
			}

		}
	}

	for _, parentLayer := range layer.parents {
		parentLayer.Backward(backwardGradients)
	}
}

func (layer *Linear) GetUpdatables() []Updatable {
	updatables := []Updatable{
		layer.weights,
		layer.biases,
	}
	return updatables
}

func (layer Linear) String() string {
	stringified := "Linear Layer:"

	stringified += "\n\tweights: " + fmt.Sprint(layer.weights.Values)
	stringified += "\n\tbiases: " + fmt.Sprint(layer.biases.Values)

	return stringified
}
