package main

import (
	. "djanngo/v3/layers"
	"math"
)

type Loss interface {
	Init(Network)
	ComputeLoss([][]float64, [][]float64) []float64
	ComputeGradients([][]float64, [][]float64) [][]float64
}

type MSELoss struct {
	previousLayer Layer
}

func (mseloss *MSELoss) Init(network Network) {
	mseloss.previousLayer = network.layers[len(network.layers)-1]
}

func (mseloss *MSELoss) ComputeLoss(prediction [][]float64, target [][]float64) float64 {
	loss := 0.0

	for i := 0; i < len(prediction); i++ {

		for j := 0; j < len(prediction[i]); j++ {
			loss += 1 / float64(len(prediction)) * 0.5 * math.Pow(prediction[i][j]-target[i][j], 2)
		}
	}

	return loss
}

func (mseloss *MSELoss) ComputeGradients(prediction [][]float64, target [][]float64) [][]float64 {
	lossGradients := make([][]float64, len(prediction))

	for i := 0; i < len(prediction); i++ {

		lossGradients[i] = make([]float64, len(prediction[i]))

		for j := 0; j < len(prediction[i]); j++ {
			lossGradients[i][j] = 1 / float64(len(prediction)) * (prediction[i][j] - target[i][j])
		}
	}

	return lossGradients
}
