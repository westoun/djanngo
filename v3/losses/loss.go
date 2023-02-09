package losses

import (
	. "djanngo/v3/networks"
)

type Loss interface {
	Init(Network)
	ComputeLoss([][]float64, [][]float64) []float64
	ComputeGradients([][]float64, [][]float64) [][]float64
}
