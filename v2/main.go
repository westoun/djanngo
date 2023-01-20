package main

func main() {
	layers := []int{2, 5, 1}
	activationFunction := ReLU{}
	lossFunction := MSE{}

	optimizer := MomentumGD{}
	optimizer.init(0.9)

	scheduler := LossDecreaseScheduler{}
	scheduler.init(0.01, 0.1, 0)

	network := ANN{}
	network.init(layers, activationFunction)

	x := [][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
		{0.0, 0.0},
		{1.0, 1.0},
	}
	y := [][]float64{
		{1.0},
		{1.0},
		{0.0},
		{1.0},
	}

	train(&network, x, y, 50, lossFunction, &optimizer, &scheduler)
}
