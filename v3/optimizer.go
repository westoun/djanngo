package main

import . "djanngo/v3/updatables"

type Optimizer interface {
	Optimize(Network)
}

type SGD struct {
}

func (sgd *SGD) Optimize(network Network) {
	lr := 0.05

	for _, layer := range network.layers {
		for _, updatable := range layer.GetUpdatables() {
			updatableVector, isVector := updatable.(UpdatableVector)

			if isVector {
				// TODO: Move update logic to separate function
				for g := 0; g < len(updatableVector.Gradients[0]); g++ {
					updatableVector.Values[g] -= lr * updatableVector.Gradients[0][g]
				}

				updatableVector.Gradients = make([][]float64, 0)

			} else {
				updatableMatrix := updatable.(UpdatableMatrix)

				for r := 0; r < len(updatableMatrix.Gradients[0]); r++ {
					for c := 0; c < len(updatableMatrix.Gradients[0][r]); c++ {

						updatableMatrix.Values[r][c] -= lr * updatableMatrix.Gradients[0][r][c]

					}
				}

				updatableMatrix.Gradients = make([][][]float64, 0)
			}

		}
	}

}
