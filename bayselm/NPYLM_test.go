package bayselm

import (
	"math/rand"
	"testing"
	"time"
)

func TestPerformanceOfNPYLM(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	var theta float64
	var d float64
	var base float64
	var epoch int
	var maxN int
	var alpha float64
	var beta float64
	alpha = 1.0
	beta = 1.0
	maxN = 2
	base = float64(1.0 / 2097152.0) // 1 / 2^21 , size of character vocabulary in utf-8 encodeing
	theta = 1.0
	d = 0.1
	epoch = 5
	var hpylm NgramLM
	hpylm = NewHPYLM(maxN-1, theta, d, 1.0, 1.0, 1.0, 1.0, base)
	var npylm NgramLM
	npylm = NewNPYLM(theta, d, 1.0, 1.0, 1.0, 1.0, alpha, beta, maxN, 30)

	dataContainerForTrain := NewDataContainerFromAnnotatedData("../alice.train.txt")
	dataContainerForTest := NewDataContainerFromAnnotatedData("../alice.test.txt")
	for e := 0; e < epoch; e++ {
		hpylm.Train(dataContainerForTrain)
		npylm.Train(dataContainerForTrain)
	}

	perplexityOfHpylm := CalcPerplexity(hpylm, dataContainerForTest)
	perplexityOfNpylm := CalcPerplexity(npylm, dataContainerForTest)
	if !(perplexityOfNpylm < perplexityOfHpylm) {
		t.Error("probably error! a perplexity of NPYLM is expected to be lower than a perplexity of HPYLM. ", "perplexityOfNpylm = ", perplexityOfNpylm, "perplexityOfHpylm = ", perplexityOfHpylm)
	}
}
