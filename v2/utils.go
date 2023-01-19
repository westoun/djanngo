package main

import "fmt"

func sum(values []float64) float64 {
	total := 0.0

	for _, value := range values {
		total += value
	}

	return total
}

func createEmptyCopy(array interface{}) interface{} {
	copy1, has1dim := array.([]float64)
	if has1dim {

		for i := 0; i < len(copy1); i++ {
			copy1[i] = 0.0
		}

		return copy1
	}

	copy2, has2dim := array.([][]float64)
	if has2dim {

		for i := 0; i < len(copy2); i++ {
			for j := 0; j < len(copy2[i]); j++ {
				copy2[i][j] = 0.0

			}
		}

		return copy2
	}

	copy3, has3dim := array.([][][]float64)
	if has3dim {
		for i := 0; i < len(copy3); i++ {
			for j := 0; j < len(copy3[i]); j++ {
				for k := 0; k < len(copy3[i][j]); k++ {
					copy3[i][j][k] = 0.0
				}
			}
		}

		return copy3
	}

	errorMessage := fmt.Sprintf("No casting has been implemented for %T", array)
	panic(errorMessage)

}
