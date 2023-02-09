package main

import (
	. "djanngo/v3/layers"
	. "djanngo/v3/losses"
	. "djanngo/v3/networks"
	. "djanngo/v3/optimizers"
	"fmt"
)

func main() {
	network := Network{}

	linear1 := &Linear{}
	linear1.Init(2, 2, []Layer{})

	sigmoid1 := &Sigmoid{}
	sigmoid1.Init([]Layer{linear1})

	linear2 := &Linear{}
	linear2.Init(2, 1, []Layer{sigmoid1})

	sigmoid2 := &Sigmoid{}
	sigmoid2.Init([]Layer{linear2})

	network.Layers = []Layer{
		linear1,
		sigmoid1,
		linear2,
		sigmoid2,
	}

	loss := MSELoss{}

	optimizer := SGD{}

	x := [][]float64{
		{
			1.0,
			0.0,
		},
		{
			0.0,
			1.0,
		},
		{
			1.0,
			1.0,
		},
		{
			0.0,
			0.0,
		},
	}
	x = normalize(x)

	y := [][]float64{
		{
			1.0,
		},
		{
			1.0,
		},
		{
			0.0,
		},
		{
			0.0,
		},
	}

	learningRate := 0.01

	epochs := 1000
	for epoch := 1; epoch <= epochs; epoch++ {
		prediction := network.Predict(x)

		currentLoss := loss.ComputeLoss(prediction, y)
		lossGradients := loss.ComputeGradients(prediction, y)

		network.UpdateGradients(lossGradients)

		optimizer.Optimize(network, learningRate)

		if epoch%100 == 0 {
			fmt.Println("Loss after epoch ", epoch, " : ", currentLoss)
		}
	}

	fmt.Println(network)

	x = [][]float64{
		{
			1.0,
			0.0,
		},
		{
			1.0,
			1.0,
		},
	}
	predictions := network.Predict(x)
	fmt.Println(predictions)

}
