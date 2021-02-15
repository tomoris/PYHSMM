package bayselm

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestNPYLM(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	var theta float64
	var d float64
	var epoch int
	var maxN int
	var batch int
	var threads int
	var alpha float64
	var beta float64
	alpha = 1.0
	beta = 1.0
	maxN = 2
	theta = 1.0
	d = 0.1
	epoch = 2
	batch = 128
	threads = 1
	npylm := NewNPYLM(theta, d, 1.0, 1.0, 1.0, 1.0, alpha, beta, maxN, 1, "")

	dataContainerForTrain := NewDataContainer("../data/alice.raw", "", 128)
	npylm.Initialize(dataContainerForTrain)
	for e := 0; e < epoch; e++ {
		npylm.TrainWordSegmentation(dataContainerForTrain, threads, batch)
	}
	for i := 0; i < dataContainerForTrain.Size; i++ {
		npylm.removeWordSeqAsCustomer(dataContainerForTrain.SamplingWordSeqs[i])
	}
	if !(len(npylm.restaurants) == 0) {
		t.Error("len(npylm.restaurants) is not 0", npylm.restaurants)
	}
	if !(len(npylm.word2sampledDepthMemory) == 0) {
		t.Error("len(word2sampledDepthMemory) is not 0", npylm.word2sampledDepthMemory)
	}
}

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
	charVocabSize := 100.0     // 適当
	averageLengthOfWord := 6.0 // 適当
	base = float64(1.0 / math.Pow(charVocabSize, averageLengthOfWord))
	theta = 1.0
	d = 0.1
	epoch = 5
	var hpylm NgramLM
	hpylm = NewHPYLM(maxN-1, theta, d, 1.0, 1.0, 1.0, 1.0, base)
	var npylm NgramLM
	npylm = NewNPYLM(theta, d, 1.0, 1.0, 1.0, 1.0, alpha, beta, maxN, 30, "")

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
