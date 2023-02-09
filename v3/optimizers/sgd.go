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
				for g := 0; g < len(updatableVector.Gradients); g++ {
					updatableVector.Values[g] -= learningRate * updatableVector.Gradients[g]
				}

			} else {
				updatableMatrix := updatable.(UpdatableMatrix)

				for r := 0; r < len(updatableMatrix.Gradients); r++ {
					for c := 0; c < len(updatableMatrix.Gradients[r]); c++ {

						updatableMatrix.Values[r][c] -= learningRate * updatableMatrix.Gradients[r][c]

					}
				}
			}

		}
	}

}
