package main

func sum(values []float64) float64 {
	total := 0.0

	for _, value := range values {
		total += value
	}

	return total
}
