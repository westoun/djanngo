package layers

import . "djanngo/v3/updatables"

// decision: everything batch wise
type Layer interface {
	Forward([][]float64) [][]float64
	Backward([][]float64)
	GetUpdatables() []Updatable
}
