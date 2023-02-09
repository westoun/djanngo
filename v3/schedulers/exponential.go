package schedulers

import "math"

type ExponentialDecayScheduler struct {
	initialLearningRate float64
	decayRate           float64
	decaySteps          int
}

func (scheduler *ExponentialDecayScheduler) Init(initialLearningRate float64,
	decayRate float64, decaySteps int) {
	scheduler.initialLearningRate = initialLearningRate
	scheduler.decayRate = decayRate
	scheduler.decaySteps = decaySteps
}

func (scheduler *ExponentialDecayScheduler) GetLearningRate(epoch int) float64 {
	return scheduler.initialLearningRate * math.Pow(scheduler.decayRate,
		float64(epoch)/float64(scheduler.decaySteps))
}
