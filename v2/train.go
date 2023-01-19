package main

import "fmt"

func train(ann *ANN, valueBatch [][]float64, batchTargets [][]float64, epochs int, lossFunction LossFunction) {
	learningRate := 0.1

	for epoch := 1; epoch <= epochs; epoch++ {

		epochLoss := 0.0

		batchCostDeltas := make([][]float64, len(valueBatch))

		batchPredictions := ann.predict(valueBatch)
		for i, prediction := range batchPredictions {
			target := batchTargets[i]

			epochLoss += lossFunction.compute(prediction, target)
			batchCostDeltas[i] = lossFunction.derive(prediction, target)
		}

		totalWeightDeltas, totalBiasDeltas := ann.backwards(valueBatch, batchCostDeltas)

		ann.update(totalWeightDeltas, totalBiasDeltas, learningRate)

		if epoch%10 == 0 {
			fmt.Printf("Total loss in epoch %v: %v\n", epoch, epochLoss)
		}

		if epoch > 49 {
			learningRate *= 0.1
		}
	}

}
