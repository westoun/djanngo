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

func (momentum *Momentum) updateVector(updatableVector *UpdatableVector, index int, learningRate float64) {
	var v []float64
	if momentum.previousVectorVs[index] == nil {
		v = DeepCopy(updatableVector.Gradients).([]float64)
	} else {
		v = DeepCopy(momentum.previousVectorVs[index]).([]float64)

		for i := 0; i < len(v); i++ {
			v[i] = momentum.beta*v[i] + (1-momentum.beta)*updatableVector.Gradients[i]
		}
	}
	momentum.previousVectorVs[index] = v

	for g := 0; g < len(v); g++ {
		updatableVector.Values[g] -= learningRate * v[g]
	}
}

func (momentum *Momentum) updateMatrix(updatableMatrix *UpdatableMatrix, index int, learningRate float64) {
	var v [][]float64
	if momentum.previousMatrixVs[index] == nil {
		v = DeepCopy(updatableMatrix.Gradients).([][]float64)
	} else {
		v = DeepCopy(momentum.previousMatrixVs[index]).([][]float64)

		for i := 0; i < len(v); i++ {
			for j := 0; j < len(v[i]); j++ {
				v[i][j] = momentum.beta*v[i][j] + (1-momentum.beta)*updatableMatrix.Gradients[i][j]
			}
		}
	}
	momentum.previousMatrixVs[index] = v

	for r := 0; r < len(v); r++ {
		for c := 0; c < len(v[r]); c++ {
			updatableMatrix.Values[r][c] -= learningRate * v[r][c]
		}
	}

}

func (momentum *Momentum) Optimize(network Network, learningRate float64) {
	updatableIndex := 0
	for _, layer := range network.Layers {
		for _, updatable := range layer.GetUpdatables() {
			updatableVector, isVector := updatable.(UpdatableVector)

			if isVector {
				momentum.updateVector(&updatableVector, updatableIndex, learningRate)
			} else {
				updatableMatrix := updatable.(UpdatableMatrix)
				momentum.updateMatrix(&updatableMatrix, updatableIndex, learningRate)
			}

			updatableIndex++
		}
	}

}
