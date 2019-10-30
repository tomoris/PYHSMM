package bayselm

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestPYHSMM(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	var theta float64
	var d float64
	var epoch int
	var maxN int
	var batch int
	var threads int
	var alpha float64
	var beta float64
	var posSize int
	alpha = 1.0
	beta = 1.0
	maxN = 2
	theta = 1.0
	d = 0.1
	epoch = 2
	batch = 128
	threads = 1
	posSize = 1
	pyhsmm := NewPYHSMM(theta, d, 1.0, 1.0, 1.0, 1.0, alpha, beta, maxN, 8, posSize)

	dataContainerForTrain := NewDataContainer("../data/alice.raw")
	pyhsmm.Initialize(dataContainerForTrain)
	for e := 0; e < epoch; e++ {
		pyhsmm.TrainWordSegmentation(dataContainerForTrain, threads, batch)
	}
	for i := 0; i < dataContainerForTrain.Size; i++ {
		pyhsmm.removeWordSeqAsCustomer(dataContainerForTrain.SamplingWordSeqs[i], dataContainerForTrain.SamplingPosSeqs[i])
	}
	for i := 0; i < posSize+1; i++ {
		if !(len(pyhsmm.npylms[i].restaurants) == 0) {
			t.Error("len(npylm.restaurants) is not 0", pyhsmm.npylms[0].restaurants)
		}
		if !(len(pyhsmm.npylms[i].word2sampledDepthMemory) == 0) {
			t.Error("len(word2sampledDepthMemory) is not 0", pyhsmm.npylms[i].word2sampledDepthMemory)
		}
	}
}
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
