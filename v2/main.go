package main

import (
	"fmt"
	"math/rand"
)

func generateData(size int) ([][]float64, [][]float64) {
	x := make([][]float64, size)
	y := make([][]float64, size)

	for i := 0; i < size; i++ {

		min := -100.0
		max := 100.0

		x1 := rand.Float64()*(max-min) + min
		x2 := rand.Float64()*(max-min) + min

		y1 := float64(int(x1+x2) % 2)

		x[i] = []float64{
			x1, x2,
		}

		y[i] = []float64{y1}
	}

	return x, y
}

func augmentData(x [][]float64, y [][]float64, factor int) ([][]float64, [][]float64) {
	augmentedX := make([][]float64, len(x)*factor)
	augmentedY := make([][]float64, len(x)*factor)

	for i := 0; i < len(x); i++ {

		for r := 0; r < factor; r++ {
			augmentedX[factor*i+r] = make([]float64, len(x[i]))
			augmentedY[factor*i+r] = y[i]

			for j := 0; j < len(x[i]); j++ {
				augmentedX[factor*i+r][j] = x[i][j] + rand.Float64()*0.02 - 0.01
			}
		}
	}

	return augmentedX, augmentedY
}

func main() {
	layers := []int{2, 2, 2, 2, 2, 2, 1}
	activationFunction := Sigmoid{}
	lossFunction := MSE{}

	optimizer := GradientDescent{}
	// optimizer := MomentumGD{}
	// optimizer.init(0.9)

	scheduler := ConstantScheduler{}
	scheduler.init(1)
	// scheduler := LossDecreaseScheduler{}
	// scheduler.init(0.01, 0.5, 0.001)

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
	// x, y = augmentData(x, y, 100)

	// x, y := generateData(1000)

	fmt.Println("\n", network)
	train(&network, x, y, 10, lossFunction, &optimizer, &scheduler)
	fmt.Println("\n", network.weights)
	fmt.Println("\n", network.biases)

	// x = [][]float64{
	// 	{1.0, 2.0},
	// 	{20.0, -30.0},
	// 	{-1.0, -1.0},
	// }
	x = [][]float64{
		{1.0, 0.0},
		{1.0, 1.0},
		{0.0, 0.0},
		{0.0, 1.0},
	}
	prediction := network.predict(x)
	fmt.Println("\n", prediction)

}
