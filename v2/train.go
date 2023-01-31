package main

import (
	"fmt"
)

func train(ann *ANN, valueBatch [][]float64, batchTargets [][]float64, epochs int,
	lossFunction LossFunction, optimizer Optimizer, lrScheduler LRScheduler) {

	for epoch := 1; epoch <= epochs; epoch++ {

		epochLoss := 0.0

		batchCostDeltas := make([][]float64, len(valueBatch))

		batchPredictions := ann.predict(valueBatch)

		for i, prediction := range batchPredictions {
			target := batchTargets[i]

			epochLoss += 1 / float64(len(batchPredictions)) * lossFunction.compute(prediction, target)

			batchCostDeltas[i] = lossFunction.derive(prediction, target)
		}

		batchCostDeltas = optimizer.optimize(batchCostDeltas)

		totalWeightDeltas, totalBiasDeltas := ann.backwards(valueBatch, batchCostDeltas)

		learningRate := lrScheduler.get(epoch, epochLoss)

		ann.update(totalWeightDeltas, totalBiasDeltas, learningRate)

		if epoch%100 == 0 {
			fmt.Printf("Total loss in epoch %v: %v\n", epoch, epochLoss)
		}
	}

}
