package main

func relu(val float64) float64 {
	if val <= 0 {
		return 0.0
	}

	return val
}

func reluDerivative(val float64) float64 {
	if val <= 0 {
		return 0.0
	}

	return 1.0
}

func sum(values []float64) float64 {
	total := 0.0

	for _, value := range values {
		total += value
	}

	return total
}
