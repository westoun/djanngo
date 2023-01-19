package main

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
