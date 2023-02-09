package updatables

type UpdatableMatrix struct {
	Values    [][]float64
	Gradients [][][]float64
}

func (updatableMatrix *UpdatableMatrix) ResetGrad() {
	updatableMatrix.Gradients = make([][][]float64, 0)
}
