package main

import (
	. "djanngo/v3/layers"
	"fmt"
)

func main() {
	network := Network{}

	linear1 := &Linear{}
	linear1.Init(2, 2, []Layer{})

	sigmoid1 := &Sigmoid{}
	sigmoid1.Init(2, []Layer{linear1})

	linear2 := &Linear{}
	linear2.Init(2, 1, []Layer{sigmoid1})

	sigmoid2 := &Sigmoid{}
	sigmoid2.Init(1, []Layer{linear2})

	network.layers = []Layer{
		linear1,
		sigmoid1,
		linear2,
		sigmoid2,
	}

	loss := MSELoss{}
	loss.Init(network)

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

	epochs := 100
	for epoch := 1; epoch <= epochs; epoch++ {
		prediction := network.Predict(x)

		currentLoss := loss.ComputeLoss(prediction, y)
		lossGradients := loss.ComputeGradients(prediction, y)

		network.UpdateGradients(lossGradients)

		optimizer.Optimize(network)

		if epoch%10 == 0 {
			fmt.Println("Loss after epoch ", epoch, " : ", currentLoss)
		}
	}

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
