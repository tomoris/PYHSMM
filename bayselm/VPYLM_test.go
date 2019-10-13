package bayselm

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
)

func TestVPYLM(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	var theta float64
	var d float64
	var base float64
	var epoch int
	sampledDepthMemory := make([]int, 0, 0)
	base = 1.0 / 10.0
	theta = 1.0
	d = 0.1
	epoch = 1000
	const maxDepth int = 3
	const alpha float64 = 1.0
	const beta float64 = 1.0
	vpylm := NewVPYLM(maxDepth, theta, d, 1.0, 1.0, 1.0, 1.0, base, alpha, beta)

	var word string
	word = "abc"
	u := context{"fgh", "de"}

	stopProbs := make([]float64, len(u)+1, len(u)+1)
	vpylm.calcStopProbs(u, stopProbs)
	stopProbsCorrect := []float64{0.5, 0.5, 0.5}
	if !(reflect.DeepEqual(stopProbs, stopProbsCorrect)) {
		t.Error("stopProbs = ", stopProbs, "stopProbsCorrect = ", stopProbsCorrect)
	}

	pAddZero, _, _ := vpylm.CalcProb(word, u)
	pAddZeroCorrect := (base * 0.5) + (base * 0.25) + (base * 0.125)
	if !(pAddZero == float64(pAddZeroCorrect)) {
		t.Error("pAddZero = ", pAddZero, "pAddZeroCorrect = ", pAddZeroCorrect)
	}

	body := (1.0 - d*1.0) / (theta + 1.0)
	smoothing := (theta + d*1.0) / (theta + 1.0)
	pCorrect := ((body + smoothing*(body+smoothing*(body+smoothing*base))) * (2.0 / 3.0) * (1.0 - (1.0 / 3.0)) * (1.0 - (1.0 / 3.0))) + ((body + smoothing*(body+smoothing*base)) * (1.0 / 3.0) * (1.0 - (1.0 / 3.0))) + ((body + smoothing*base) * (1.0 / 3.0))

	sampledDepth := vpylm.AddCustomer(word, u)
	sampledDepthMemory = append(sampledDepthMemory, sampledDepth)
	pAddOne, probsAddOne, _ := vpylm.CalcProb(word, u)
	if !(math.Abs(float64((pAddOne - float64(pCorrect)))) < 0.00001) {
		t.Error("Maybe error! please test several times. ", "pAddOne = ", pAddOne, "pCorrect = ", pCorrect)
	}
	if !(pAddOne >= pAddZero) {
		t.Error("pAddOne = ", pAddOne, "pAddZero = ", pAddZero)
	}

	for i := 0; i < epoch; i++ {
		sampledDepth := vpylm.AddCustomer(word, u)
		sampledDepthMemory = append(sampledDepthMemory, sampledDepth)
	}
	pAddMany, probsAddMany, _ := vpylm.CalcProb(word, u)
	if !(pAddMany >= pAddOne) {
		t.Error("pAddMany = ", pAddMany, "pAddOne = ", pAddOne)
	}
	for i := 0; i < len(probsAddMany); i++ {
		if !(probsAddMany[i] >= probsAddOne[i]) {
			t.Error("probsAddMany[i] = ", probsAddMany[i], "probsAddOne[i] = ", probsAddOne[i], "i = ", i)
		}
	}

	for i := 0; i < epoch; i++ {
		vpylm.RemoveCustomer(word, u, sampledDepthMemory[i+1])
	}
	pRemoveMany, probsRemoveMany, _ := vpylm.CalcProb(word, u)
	if !(pRemoveMany == pAddOne) {
		t.Error("pRemoveMany = ", pRemoveMany, "pAddOne = ", pAddOne)
	}
	for i := 0; i < len(probsRemoveMany); i++ {
		if !(probsRemoveMany[i] == probsAddOne[i]) {
			t.Error("probsRemoveMany[i] = ", probsRemoveMany[i], "probsAddOne[i] = ", probsAddOne[i], "i = ", i)
		}
	}

	vpylm.RemoveCustomer(word, u, sampledDepthMemory[0])
	pRemoveOne, _, _ := vpylm.CalcProb(word, u)
	if !(pRemoveOne == pAddZero) {
		t.Error("pRemoveOne = ", pRemoveOne, "pAddZero = ", pAddZero)
	}

	if !(len(vpylm.hpylm.restaurants) == 0) {
		t.Error("vpylm.restaurants = ", vpylm.hpylm.restaurants)
	}
}

func TestPerformanceOfVPYLM(t *testing.T) {
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
	maxN = 3
	base = float64(1.0 / 2097152.0) // 1 / 2^21 , size of character vocabulary in utf-8 encodeing
	theta = 1.0
	d = 0.1
	epoch = 2000
	var hpylm NgramLM
	hpylm = NewHPYLM(maxN-1, theta, d, 1.0, 1.0, 1.0, 1.0, base)
	var vpylm NgramLM
	vpylm = NewVPYLM(maxN-1, theta, d, 1.0, 1.0, 1.0, 1.0, base, alpha, beta)

	dataContainerForTrain := NewDataContainerFromAnnotatedData("../alice.train.txt")
	dataContainerForTest := NewDataContainerFromAnnotatedData("../alice.test.txt")
	for e := 0; e < epoch; e++ {
		hpylm.Train(dataContainerForTrain)
		vpylm.Train(dataContainerForTrain)
	}

	perplexityOfHpylm := CalcPerplexity(hpylm, dataContainerForTest)
	perplexityOfVpylm := CalcPerplexity(vpylm, dataContainerForTest)
	if !(perplexityOfVpylm < perplexityOfHpylm) {
		t.Error("probably error! a perplexity of VPYLM is expected to be lower than a perplexity of HPYLM. ", "perplexityOfVpylm = ", perplexityOfVpylm, "perplexityOfHpylm = ", perplexityOfHpylm)
	}
}
