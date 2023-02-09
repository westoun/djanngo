package optimizers

import (
	. "djanngo/v3/networks"
)

type Optimizer interface {
	Optimize(Network, float64)
}
