package optimizers

import (
	. "djanngo/v3/networks"
	. "djanngo/v3/updatables"
)

type SGD struct {
}

func (sgd *SGD) Optimize(network Network, learningRate float64) {
	for _, layer := range network.Layers {
		for _, updatable := range layer.GetUpdatables() {
			updatableVector, isVector := updatable.(UpdatableVector)

			if isVector {
				// TODO: Move update logic to separate function
				for g := 0; g < len(updatableVector.Gradients[0]); g++ {
					updatableVector.Values[g] -= learningRate * updatableVector.Gradients[0][g]
				}

				updatableVector.ResetGrad()

			} else {
				updatableMatrix := updatable.(UpdatableMatrix)

				for r := 0; r < len(updatableMatrix.Gradients[0]); r++ {
					for c := 0; c < len(updatableMatrix.Gradients[0][r]); c++ {

						updatableMatrix.Values[r][c] -= learningRate * updatableMatrix.Gradients[0][r][c]

					}
				}

				updatableMatrix.ResetGrad()
			}

		}
	}

}
