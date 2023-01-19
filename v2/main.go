package main

func main() {
	layers := []int{2, 2, 1}

	network := ANN{}
	network.init(layers)

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

	train(&network, x, y, 50)
}
