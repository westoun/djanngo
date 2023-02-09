package optimizers

import (
	. "djanngo/v3/networks"
	. "djanngo/v3/updatables"
	. "djanngo/v3/utils"
)

type Momentum struct {
	beta             float64
	previousVectorVs map[int][]float64
	previousMatrixVs map[int][][]float64
}

func (momentum *Momentum) Init(beta float64) {
	momentum.beta = beta
	momentum.previousVectorVs = make(map[int][]float64)
	momentum.previousMatrixVs = make(map[int][][]float64)
}

func (momentum *Momentum) Optimize(network Network, learningRate float64) {

	updatableIndex := 0
	for _, layer := range network.Layers {
		for _, updatable := range layer.GetUpdatables() {
			updatableVector, isVector := updatable.(UpdatableVector)

			if isVector {
				// TODO: Move update logic to separate function

				var v []float64
				if momentum.previousVectorVs[updatableIndex] == nil {
					v = DeepCopy(updatableVector.Gradients[0]).([]float64)
				} else {
					v = DeepCopy(momentum.previousVectorVs[updatableIndex]).([]float64)

					for i := 0; i < len(v); i++ {
						v[i] = momentum.beta*v[i] + (1-momentum.beta)*updatableVector.Gradients[0][i]
					}
				}
				momentum.previousVectorVs[updatableIndex] = v

				for g := 0; g < len(v); g++ {
					updatableVector.Values[g] -= learningRate * v[g]
				}

				updatableVector.Gradients = make([][]float64, 0)

			} else {
				updatableMatrix := updatable.(UpdatableMatrix)

				var v [][]float64
				if momentum.previousMatrixVs[updatableIndex] == nil {
					v = DeepCopy(updatableMatrix.Gradients[0]).([][]float64)
				} else {
					v = DeepCopy(momentum.previousMatrixVs[updatableIndex]).([][]float64)

					for i := 0; i < len(v); i++ {
						for j := 0; j < len(v[i]); j++ {
							v[i][j] = momentum.beta*v[i][j] + (1-momentum.beta)*updatableMatrix.Gradients[0][i][j]
						}
					}
				}
				momentum.previousMatrixVs[updatableIndex] = v

				for r := 0; r < len(v); r++ {
					for c := 0; c < len(v[r]); c++ {

						updatableMatrix.Values[r][c] -= learningRate * v[r][c]

					}
				}

				updatableMatrix.Gradients = make([][][]float64, 0)
			}

			updatableIndex++
		}
	}

}
