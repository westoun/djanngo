package layers

import (
	. "djanngo/v3/updatables"
	"math"
)

type Sigmoid struct {
	parents   []Layer
	lastBatch [][]float64
	layerSize int
}

func (layer *Sigmoid) Init(layerSize int, parents []Layer) {
	layer.layerSize = layerSize
	layer.parents = parents
}

func (layer *Sigmoid) Forward(batch [][]float64) [][]float64 {

	layer.lastBatch = batch

	output := make([][]float64, len(batch))

	for i, values := range batch {
		output[i] = make([]float64, len(values))

		for j, value := range values {
			output[i][j] = 1 / (1 + math.Exp(-value))
		}
	}

	return output
}

func (layer *Sigmoid) Backward(incomingGradients [][]float64) {
	backwardGradients := make([][]float64, len(incomingGradients))

	// assert same length
	for i := 0; i < len(layer.lastBatch); i++ {
		backwardGradients[i] = make([]float64, len(layer.lastBatch[i]))

		for j := 0; j < len(layer.lastBatch[i]); j++ {
			lastValue := layer.lastBatch[i][j]

			backwardGradients[i][j] =
				incomingGradients[i][j] *
					1 / (1 + math.Exp(-lastValue)) *
					(1 - 1/(1+math.Exp(-lastValue)))

		}
	}

	for _, parentLayer := range layer.parents {
		parentLayer.Backward(backwardGradients)
	}

}

func (sigmoid *Sigmoid) GetUpdatables() []Updatable {
	return []Updatable{}
}
