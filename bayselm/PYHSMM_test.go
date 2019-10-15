package bayselm

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestPerformanceOfPYHSMM(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	var theta float64
	var d float64
	var epoch int
	var maxN int
	var alpha float64
	var beta float64
	var posSize int
	alpha = 1.0
	beta = 1.0
	maxN = 2
	theta = 1.0
	d = 0.1
	epoch = 5
	posSize = 4
	var npylm NgramLM
	npylm = NewNPYLM(theta, d, 1.0, 1.0, 1.0, 1.0, alpha, beta, maxN, 8)
	var pyhsmm NgramLM
	pyhsmm = NewPYHSMM(theta, d, 1.0, 1.0, 1.0, 1.0, alpha, beta, maxN, 8, posSize)

	dataContainerForTrain := NewDataContainerFromAnnotatedData("../alice.train.txt")
	dataContainerForTest := NewDataContainerFromAnnotatedData("../alice.test.txt")
	for e := 0; e < epoch; e++ {
		fmt.Println("epoch = ", e)
		pyhsmm.Train(dataContainerForTrain)
		npylm.Train(dataContainerForTrain)
	}

	npylmDummy := NewNPYLM(theta, d, 1.0, 1.0, 1.0, 1.0, alpha, beta, maxN, 8)
	npylmDummy.InitializeFromAnnotatedData(dataContainerForTest.Sents, dataContainerForTest.SamplingWordSeqs)
	perplexityOfNpylm := CalcPerplexity(npylm, dataContainerForTest)
	perplexityOfPyhsmm := CalcPerplexity(pyhsmm, dataContainerForTest)
	if !(perplexityOfPyhsmm < perplexityOfNpylm) {
		t.Error("probably error! a perplexity of NPYLM is expected to be lower than a perplexity of HPYLM. ", "perplexityOfPyhsmm = ", perplexityOfPyhsmm, "perplexityOfNpylm = ", perplexityOfNpylm)
	}
}
