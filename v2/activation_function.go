package main

import "math"

type ActivationFunction interface {
	compute(val float64) float64
	derive(val float64) float64
}

type ReLU struct{}

func (_ ReLU) compute(val float64) float64 {
	if val <= 0 {
		return 0.0
	}

	return val
}

func (_ ReLU) derive(val float64) float64 {
	if val <= 0 {
		return 0.0
	}

	return 1.0
}

type Sigmoid struct{}

func (_ Sigmoid) compute(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

func (s Sigmoid) derive(val float64) float64 {
	return s.compute(val) * (1 - s.compute(val))
}
