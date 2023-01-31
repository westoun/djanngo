package main

import (
	"math"
)

type LossFunction interface {
	compute(prediction []float64, target []float64) float64
	derive(prediction []float64, target []float64) []float64
}

type MSE struct{}

func (_ MSE) compute(prediction []float64, target []float64) float64 {
	loss := 0.0

	for i := 0; i < len(prediction); i++ {
		loss += 1 / float64(len(prediction)) * math.Pow((prediction[i]-target[i]), 2)
	}

	return loss
}

func (_ MSE) derive(prediction []float64, target []float64) []float64 {
	derivative := make([]float64, len(prediction))

	for i := 0; i < len(prediction); i++ {
		derivative[i] = 2 / float64(len(prediction)) * (prediction[i] - target[i])
	}

	return derivative
}

type MSELogScaled struct{}

func (_ MSELogScaled) compute(prediction []float64, target []float64) float64 {
	// taken from https://stats.stackexchange.com/a/519735

	loss := 0.0

	for i := 0; i < len(prediction); i++ {
		loss += 1/float64(len(prediction))*math.Pow((prediction[i]-target[i]), 2) +
			5*1/float64(len(prediction))*math.Pow((math.Log(prediction[i]+10e-9)-math.Log(target[i]+10e-9)), 2)
	}

	return loss
}

func (_ MSELogScaled) derive(prediction []float64, target []float64) []float64 {
	derivative := make([]float64, len(prediction))

	for i := 0; i < len(prediction); i++ {
		derivative[i] = 2/float64(len(prediction))*(prediction[i]-target[i]) +
			5*2/float64(len(prediction))*1/(prediction[i]+10e-9)*(math.Log(prediction[i]+10e-9)-math.Log(target[i]+10e-9))
	}

	return derivative

}
