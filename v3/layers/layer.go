package layers

import . "djanngo/v3/updatables"

type Layer interface {
	Forward([][]float64) [][]float64
	Backward([][]float64)
	GetUpdatables() []Updatable
	String() string
}
