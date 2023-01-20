package main

type Optimizer interface {
	optimize(batchCostDelta [][]float64) [][]float64
}

type GradientDescent struct{}

func (_ GradientDescent) optimize(batchCostDelta [][]float64) [][]float64 {
	return batchCostDelta
}

type MomentumGD struct {
	previousDeltas    [][]float64
	predecessorWeight float64
}

func (momentumGD *MomentumGD) init(predecessorWeight float64) {
	// assert weight between 1 and 0

	momentumGD.predecessorWeight = predecessorWeight
}

func (momentumGD *MomentumGD) optimize(batchCostDelta [][]float64) [][]float64 {

	if momentumGD.previousDeltas == nil {
		momentumGD.previousDeltas = batchCostDelta
		return batchCostDelta
	}

	updatedCostDelta, _ := createEmptyCopy(batchCostDelta).([][]float64)

	for i := 0; i < len(batchCostDelta); i++ {
		for j := 0; j < len(batchCostDelta[i]); j++ {

			updatedCostDelta[i][j] = momentumGD.predecessorWeight*momentumGD.previousDeltas[i][j] +
				(1-momentumGD.predecessorWeight)*batchCostDelta[i][j]
		}
	}

	momentumGD.previousDeltas = batchCostDelta
	return updatedCostDelta
}
