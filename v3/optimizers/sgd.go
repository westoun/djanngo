package optimizers

import (
	. "djanngo/v3/networks"
	. "djanngo/v3/updatables"
)

type SGD struct {
}

func (sgd *SGD) updateVector(updatableVector *UpdatableVector, learningRate float64) {
	for g := 0; g < len(updatableVector.Gradients); g++ {
		updatableVector.Values[g] -= learningRate * updatableVector.Gradients[g]
	}
}

func (sgd *SGD) updateMatrix(updatableMatrix *UpdatableMatrix, learningRate float64) {
	for r := 0; r < len(updatableMatrix.Gradients); r++ {
		for c := 0; c < len(updatableMatrix.Gradients[r]); c++ {

			updatableMatrix.Values[r][c] -= learningRate * updatableMatrix.Gradients[r][c]

		}
	}
}

func (sgd *SGD) Optimize(network Network, learningRate float64) {
	for _, layer := range network.Layers {
		for _, updatable := range layer.GetUpdatables() {
			updatableVector, isVector := updatable.(UpdatableVector)

			if isVector {
				sgd.updateVector(&updatableVector, learningRate)

			} else {
				updatableMatrix := updatable.(UpdatableMatrix)
				sgd.updateMatrix(&updatableMatrix, learningRate)
			}
		}
	}
}
