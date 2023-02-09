package utils

import (
	"fmt"
)

func CreateEmpty1DCopy(reference []float64) []float64 {
	copy := make([]float64, len(reference))

	for i := 0; i < len(reference); i++ {
		copy[i] = 0.0
	}

	return copy
}

func CreateEmpty2DCopy(reference [][]float64) [][]float64 {
	copy := make([][]float64, len(reference))

	for i := 0; i < len(reference); i++ {
		copy[i] = make([]float64, len(reference[i]))

		for j := 0; j < len(reference[i]); j++ {
			copy[i][j] = 0.0

		}
	}

	return copy
}

func CreateEmpty3DCopy(reference [][][]float64) [][][]float64 {
	copy := make([][][]float64, len(reference))

	for i := 0; i < len(reference); i++ {

		copy[i] = make([][]float64, len(reference[i]))

		for j := 0; j < len(reference[i]); j++ {

			copy[i][j] = make([]float64, len(reference[i][j]))

			for k := 0; k < len(reference[i][j]); k++ {
				copy[i][j][k] = 0.0
			}
		}
	}

	return copy
}

func CreateEmptyCopy(array interface{}) interface{} {
	reference1, has1dim := array.([]float64)
	if has1dim {
		return CreateEmpty1DCopy(reference1)
	}

	reference2, has2dim := array.([][]float64)
	if has2dim {
		return CreateEmpty2DCopy(reference2)
	}

	reference3, has3dim := array.([][][]float64)
	if has3dim {
		return CreateEmpty3DCopy(reference3)
	}

	errorMessage := fmt.Sprintf("No casting has been implemented for %T", array)
	panic(errorMessage)

}

func DeepCopy1D(reference []float64) []float64 {
	copy := make([]float64, len(reference))

	for i := 0; i < len(reference); i++ {
		copy[i] = reference[i]
	}

	return copy
}

func DeepCopy2D(reference [][]float64) [][]float64 {
	copy := make([][]float64, len(reference))

	for i := 0; i < len(reference); i++ {

		copy[i] = make([]float64, len(reference[i]))

		for j := 0; j < len(reference[i]); j++ {
			copy[i][j] = reference[i][j]
		}
	}

	return copy
}

func DeepCopy3D(reference [][][]float64) [][][]float64 {
	copy := make([][][]float64, len(reference))

	for i := 0; i < len(reference); i++ {

		copy[i] = make([][]float64, len(reference[i]))

		for j := 0; j < len(reference[i]); j++ {

			copy[i][j] = make([]float64, len(reference[i][j]))

			for k := 0; k < len(reference[i][j]); k++ {
				copy[i][j][k] = reference[i][j][k]
			}
		}
	}

	return copy
}

func DeepCopy(array interface{}) interface{} {
	reference1, has1dim := array.([]float64)
	if has1dim {
		return DeepCopy1D(reference1)
	}

	reference2, has2dim := array.([][]float64)
	if has2dim {
		return DeepCopy2D(reference2)
	}

	reference3, has3dim := array.([][][]float64)
	if has3dim {
		return DeepCopy3D(reference3)
	}

	errorMessage := fmt.Sprintf("No casting has been implemented for %T", array)
	panic(errorMessage)

}
