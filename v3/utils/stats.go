package utils

import "math"

func ComputeMean(data []float64) float64 {
	mean := 0.0

	for _, item := range data {
		mean += 1 / float64(len(data)) * item
	}

	return mean
}

func ComputeStdev(data []float64) float64 {
	mean := ComputeMean(data)

	stdev := 0.0
	for _, item := range data {
		stdev += 1 / float64(len(data)) * math.Pow(item-mean, 2)
	}

	return stdev
}

func Normalize(data [][]float64) [][]float64 {
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
	normalizedData := CreateEmptyCopy(data).([][]float64)
	for j := 0; j < len(valuesByDimension); j++ {
		mean := ComputeMean(valuesByDimension[j])
		sigma := ComputeStdev(valuesByDimension[j])

		for i := 0; i < len(data); i++ {
			normalizedData[i][j] = (data[i][j] - mean) / sigma
		}

	}

	return normalizedData
}
