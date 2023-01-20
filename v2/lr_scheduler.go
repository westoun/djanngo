package main

type LRScheduler interface {
	get(epoch int, loss float64) float64
}

type ConstantScheduler struct {
	lr float64
}

func (scheduler *ConstantScheduler) init(lr float64) {
	scheduler.lr = lr
}

func (scheduler *ConstantScheduler) get(epoch int,
	loss float64) float64 {
	return scheduler.lr
}

type FactoredDecreaseScheduler struct {
	lr     float64
	factor float64
}

func (scheduler *FactoredDecreaseScheduler) init(lr float64,
	decreaseFactor float64) {
	scheduler.lr = lr
	scheduler.factor = decreaseFactor
}

func (scheduler *FactoredDecreaseScheduler) get(epoch int,
	loss float64) float64 {

	if epoch > 1 {
		scheduler.lr = scheduler.lr * scheduler.factor
	}

	return scheduler.lr
}

type LossDecreaseScheduler struct {
	lr           float64
	previousLoss float64
	factor       float64
	epsilon      float64
}

func (scheduler *LossDecreaseScheduler) init(lr float64,
	decreaseFactor float64, epsilon float64) {
	scheduler.lr = lr
	scheduler.factor = decreaseFactor
	scheduler.epsilon = epsilon
}

func (scheduler *LossDecreaseScheduler) get(epoch int,
	loss float64) float64 {

	if scheduler.previousLoss == 0.0 {
		scheduler.previousLoss = loss
		return scheduler.lr
	}

	lossDelta := loss - scheduler.previousLoss
	lossQuotient := -1 * lossDelta / scheduler.previousLoss

	if lossDelta > 0 || lossQuotient < scheduler.epsilon {
		scheduler.lr = scheduler.lr * scheduler.factor
	}

	scheduler.previousLoss = loss
	return scheduler.lr
}
