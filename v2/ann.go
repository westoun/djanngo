package main

import (
	"fmt"
	"math"
	"math/rand"
)

type ANN struct {
	weights [][][]float64
	biases  [][]float64
}

func (ann *ANN) init(layers []int) {
	ann.weights = make([][][]float64, len(layers)-1)
	ann.biases = make([][]float64, len(layers)-1)

	for i := 1; i < len(layers); i++ {
		layerWeights := make([][]float64, layers[i])

		for j := 0; j < layers[i]; j++ {

			neuronWeights := make([]float64, layers[i-1])
			for k := 0; k < layers[i-1]; k++ {
				neuronWeights[k] = rand.Float64()
			}

			layerWeights[j] = neuronWeights
		}

		ann.weights[i-1] = layerWeights
	}

	for i := 1; i < len(layers); i++ {
		layerBiases := make([]float64, layers[i])

		for j := 0; j < layers[i]; j++ {
			layerBiases[j] = rand.Float64()
		}

		ann.biases[i-1] = layerBiases
	}

}

func (ann *ANN) forward(valueBatch [][]float64) (batchZs [][][]float64,
	batchActivations [][][]float64) {

	batchZs = make([][][]float64, len(valueBatch))
	batchActivations = make([][][]float64, len(valueBatch))

	for i, values := range valueBatch {
		zs, activations := ann.instanceForward(values)

		batchZs[i] = zs
		batchActivations[i] = activations
	}

	return batchZs, batchActivations
}

func (ann *ANN) instanceForward(values []float64) (zs [][]float64,
	activations [][]float64) {
	zs = make([][]float64, len(ann.weights))
	activations = make([][]float64, len(ann.weights))

	activations[0] = values

	for layer := 0; layer < len(ann.weights); layer++ {
		layerWeights := ann.weights[layer]
		layerBiases := ann.biases[layer]

		layerZs := make([]float64, len(layerWeights))
		layerActivations := make([]float64, len(layerWeights))

		for n := 0; n < len(ann.weights[layer]); n++ {
			neuronZ := layerBiases[n]

			for i := 0; i < len(activations[layer]); i++ {
				neuronZ += layerWeights[n][i] * activations[layer][i]
			}

			neuronActivation := relu(neuronZ)

			layerZs[n] = neuronZ
			layerActivations[n] = neuronActivation
		}

		zs[layer] = layerZs
		activations[layer] = layerActivations
	}

	return zs, activations
}

func (ann *ANN) instancePredict(values []float64) []float64 {
	_, activations := ann.instanceForward(values)

	predictions := activations[len(activations)-1]
	return predictions
}

func (ann *ANN) predict(valueBatch [][]float64) [][]float64 {
	_, batchActivations := ann.forward(valueBatch)

	predictions := make([][]float64, len(valueBatch))
	for i, _ := range valueBatch {
		predictions[i] = batchActivations[i][len(batchActivations[i])-1]
	}
	return predictions
}

func (ann *ANN) instanceBackwards(values []float64, costDeltas []float64) (weightDeltas [][][]float64,
	biasDeltas [][]float64) {

	weightDeltas = make([][][]float64, len(ann.weights))
	biasDeltas = make([][]float64, len(ann.biases))

	incomingDeltaSum := sum(costDeltas)

	previousLayerDeltas := make([]float64, len(ann.weights[len(ann.weights)-1][0]))
	for i := 0; i <= len(ann.weights[len(ann.weights)-1]); i++ {
		previousLayerDeltas[i] = incomingDeltaSum
	}

	zs, activations := ann.instanceForward(values)
	for layer := len(ann.weights) - 1; layer >= 0; layer-- {
		weightDeltas[layer] = make([][]float64, len(ann.weights[layer]))
		biasDeltas[layer] = make([]float64, len(ann.biases[layer]))

		for n, neuronWeights := range ann.weights[layer] {
			weightDeltas[layer][n] = make([]float64, len(ann.weights[layer][n]))

			for w, _ := range neuronWeights {
				weightDelta := previousLayerDeltas[n] * reluDerivative(
					zs[layer][n],
				) * activations[layer][n]

				weightDeltas[layer][n][w] = weightDelta
			}
		}

		for n, _ := range ann.biases[layer] {
			biasDelta := previousLayerDeltas[n] * reluDerivative(zs[layer][n])

			biasDeltas[layer][n] = biasDelta

		}

		currentLayerDeltas := make([]float64, len(ann.weights[layer][0]))
		for n, neuronWeights := range ann.weights[layer] {
			for w, weight := range neuronWeights {
				currentLayerDeltas[w] += previousLayerDeltas[n] * weight
			}
		}

		previousLayerDeltas = currentLayerDeltas

	}

	return weightDeltas, biasDeltas
}

func (ann *ANN) backwards(valueBatch [][]float64, batchCostDeltas [][]float64) ([][][]float64, [][]float64) {
	var totalWeightDeltas [][][]float64
	var totalBiasDeltas [][]float64

	for i := 0; i < len(valueBatch); i++ {

		values := valueBatch[i]
		costDeltas := batchCostDeltas[i]

		weightDeltas, biasDeltas := ann.instanceBackwards(values, costDeltas)

		if totalWeightDeltas == nil {
			totalWeightDeltas = weightDeltas
			totalBiasDeltas = biasDeltas
			continue
		}

		for l, layerDeltas := range weightDeltas {
			for n, neuronDeltas := range layerDeltas {
				for w, weightDelta := range neuronDeltas {
					totalWeightDeltas[l][n][w] += weightDelta
				}
			}
		}

		for l, layerDeltas := range biasDeltas {
			for n, neuronDelta := range layerDeltas {
				totalBiasDeltas[l][n] += neuronDelta
			}
		}
	}

	return totalWeightDeltas, totalBiasDeltas
}

func (ann *ANN) update(weightDeltas [][][]float64, biasDeltas [][]float64, learningRate float64) {
	for l, layerDeltas := range weightDeltas {
		for n, neuronDeltas := range layerDeltas {
			for w, weightDelta := range neuronDeltas {
				ann.weights[l][n][w] -= learningRate * weightDelta
			}
		}
	}

	for l, layerDeltas := range biasDeltas {
		for n, neuronDelta := range layerDeltas {
			ann.biases[l][n] -= learningRate * neuronDelta
		}
	}
}

func train(ann *ANN, valueBatch [][]float64, batchTargets [][]float64, epochs int) {
	learningRate := 0.1

	for epoch := 1; epoch <= epochs; epoch++ {

		epochLoss := 0.0

		batchCostDeltas := make([][]float64, len(valueBatch))

		batchPredictions := ann.predict(valueBatch)
		for i, prediction := range batchPredictions {
			batchCostDeltas[i] = make([]float64, len(prediction))

			target := batchTargets[i]

			for j := 0; j < len(prediction); j++ {

				epochLoss += 0.5 * math.Pow(prediction[j]-target[j], 2)
				batchCostDeltas[i][j] = prediction[j] - target[j]

			}
		}

		totalWeightDeltas, totalBiasDeltas := ann.backwards(valueBatch, batchCostDeltas)

		ann.update(totalWeightDeltas, totalBiasDeltas, learningRate)

		if epoch%2 == 0 {
			fmt.Printf("Total loss in epoch %v: %v\n", epoch, epochLoss)
		}

		if epoch > 20 {
			learningRate *= 0.5
		}
	}

}
