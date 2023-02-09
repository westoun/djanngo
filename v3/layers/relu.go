package layers

import (
	. "djanngo/v3/updatables"
)

type ReLU struct {
	parents   []Layer
	lastBatch [][]float64
}

func (layer *ReLU) Init(parents []Layer) {
	layer.parents = parents
}

func (layer *ReLU) Forward(batch [][]float64) [][]float64 {
	layer.lastBatch = batch

	output := make([][]float64, len(batch))

	for i, values := range batch {
		output[i] = make([]float64, len(values))

		for j, value := range values {
			if value > 0 {
				output[i][j] = value
			} else {
				output[i][j] = 0
			}

		}
	}

	return output
}

func (layer *ReLU) Backward(incomingGradients [][]float64) {
	backwardGradients := make([][]float64, len(incomingGradients))

	// assert same length
	for i := 0; i < len(layer.lastBatch); i++ {
		backwardGradients[i] = make([]float64, len(layer.lastBatch[i]))

		for j := 0; j < len(layer.lastBatch[i]); j++ {
			lastValue := layer.lastBatch[i][j]

			if lastValue > 0 {
				backwardGradients[i][j] = incomingGradients[i][j] * lastValue
			} else {
				backwardGradients[i][j] = 0
			}

		}
	}

	for _, parentLayer := range layer.parents {
		parentLayer.Backward(backwardGradients)
	}
}

func (layer *ReLU) GetUpdatables() []Updatable {
	return []Updatable{}
}

func (layer ReLU) String() string {
	return "ReLU Activation Layer"
}
