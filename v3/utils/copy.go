package utils

import (
	"fmt"
)

func CreateEmptyCopy(array interface{}) interface{} {
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
func DeepCopy(array interface{}) interface{} {
	reference1, has1dim := array.([]float64)
	if has1dim {

		copy := make([]float64, len(reference1))

		for i := 0; i < len(reference1); i++ {
			copy[i] = reference1[i]
		}

		return copy
	}

	reference2, has2dim := array.([][]float64)
	if has2dim {

		copy := make([][]float64, len(reference2))

		for i := 0; i < len(reference2); i++ {

			copy[i] = make([]float64, len(reference2[i]))

			for j := 0; j < len(reference2[i]); j++ {
				copy[i][j] = reference2[i][j]

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
					copy[i][j][k] = reference3[i][j][k]
				}
			}
		}

		return copy
	}

	errorMessage := fmt.Sprintf("No casting has been implemented for %T", array)
	panic(errorMessage)

}
