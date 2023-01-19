package main

func main() {
	layers := []int{2, 5, 1}
	activationFunction := Sigmoid{}
	lossFunction := MSE{}

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
		{0.0},
	}

	train(&network, x, y, 100, lossFunction)
}
