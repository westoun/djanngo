package updatables

type UpdatableVector struct {
	Values    []float64
	Gradients [][]float64
}

func (updatableVector *UpdatableVector) ResetGrad() {
	updatableVector.Gradients = make([][]float64, 0)
}
