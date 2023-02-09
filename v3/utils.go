package main

import (
	"fmt"
	"math"
)

func createEmptyCopy(array interface{}) interface{} {
	reference1, has1dim := array.([]float64)
	if has1dim {

		copy := make([]float64, len(reference1))

		for i := 0; i < len(reference1); i++ {
			copy[i] = 0.0
		}

		return copy
	}

	reference2, has2dim := array.([][]float64)
	if has2dim {

		copy := make([][]float64, len(reference2))

		for i := 0; i < len(reference2); i++ {

			copy[i] = make([]float64, len(reference2[i]))

			for j := 0; j < len(reference2[i]); j++ {
				copy[i][j] = 0.0

			}
		}

		return copy
	}

	reference3, has3dim := array.([][][]float64)
	if has3dim {

		copy := make([][][]float64, len(reference3))

		for i := 0; i < len(reference3); i++ {

			copy[i] = make([][]float64, len(reference3[i]))

			for j := 0; j < len(reference3[i]); j++ {

				copy[i][j] = make([]float64, len(reference3[i][j]))

				for k := 0; k < len(reference3[i][j]); k++ {
					copy[i][j][k] = 0.0
				}
			}
		}

		return copy
	}

	errorMessage := fmt.Sprintf("No casting has been implemented for %T", array)
	panic(errorMessage)

}

func computeMean(data []float64) float64 {
	mean := 0.0

	for _, item := range data {
		mean += 1 / float64(len(data)) * item
	}

	return mean
}

func computeStdev(data []float64) float64 {
	mean := computeMean(data)

	stdev := 0.0
	for _, item := range data {
		stdev += 1 / float64(len(data)) * math.Pow(item-mean, 2)
	}

	return stdev
}

func normalize(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return [][]float64{}
	}

	valuesByDimension := make([][]float64, len(data[0]))
	for j := 0; j < len(data[0]); j++ {
		valuesByDimension[j] = make([]float64, len(data))

		for i := 0; i < len(data); i++ {
			valuesByDimension[j][i] = data[i][j]
		}
	}

	// TODO: Handle sigma == 0
	normalizedData := createEmptyCopy(data).([][]float64)
	for j := 0; j < len(valuesByDimension); j++ {
		mean := computeMean(valuesByDimension[j])
		sigma := computeStdev(valuesByDimension[j])

		for i := 0; i < len(data); i++ {
			normalizedData[i][j] = (data[i][j] - mean) / sigma
		}

	}

	return normalizedData
}
