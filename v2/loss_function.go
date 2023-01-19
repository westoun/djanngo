package main

import "math"

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
