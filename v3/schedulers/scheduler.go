package schedulers

type Scheduler interface {
	GetLearningRate(int) float64
}
