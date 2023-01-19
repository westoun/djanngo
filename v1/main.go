package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

func loadData(csvPath string, yLength int) []DataPoint {
	file, err := os.Open(csvPath)
	defer file.Close()

	if err != nil {
		fmt.Println(err)
	}

	reader := csv.NewReader(file)
	records, _ := reader.ReadAll()

	data := make([]DataPoint, len(records))
	for i, record := range records {

		x := make([]float64, len(record)-yLength)
		y := make([]float64, yLength)

		for x_i, item := range record[:len(record)-yLength] {
			x[x_i], _ = strconv.ParseFloat(item, 8)
		}
		for y_i, item := range record[len(record)-yLength:] {
			y[y_i], _ = strconv.ParseFloat(item, 8)
		}

		data[i] = DataPoint{
			x: x,
			y: y,
		}
	}

	return data
}

func generateGreaterThanData(length int) []DataPoint {
	min := -100.0
	max := 100.0

	data := make([]DataPoint, length)

	for i := 0; i < length; i++ {
		val1 := rand.Float64()*(max-min) - min
		val2 := rand.Float64()*(max-min) - min

		x := []float64{val1, val2}

		y := []float64{0.0}
		if val2 > val1 {
			y = []float64{1.0}
		}

		data[i] = DataPoint{
			x: x,
			y: y,
		}
	}

	return data
}

func main() {

	// data := loadData("xor_dataset.csv", 1)
	// layers := []int{2, 1000, 1}

	data := generateGreaterThanData(1000)

	layers := []int{2, 1000, 1}

	fmt.Println("Configuration: ", layers, "\n ")

	ann := ANN{}
	ann.init(layers)

	ann.Train(data, 20)
}
