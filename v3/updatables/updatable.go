package updatables

type Updatable interface{}

type UpdatableVector struct {
	Values    []float64
	Gradients [][]float64
}

type UpdatableMatrix struct {
	Values    [][]float64
	Gradients [][][]float64
}
