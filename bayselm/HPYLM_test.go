package bayselm

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestHPYLM(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	var theta float64
	var d float64
	var base float64
	var epoch int
	base = 1.0 / 10.0
	theta = 1.0
	d = 0.1
	epoch = 1000
	hpylm := NewHPYLM(2, theta, d, 1.0, 1.0, 1.0, 1.0, base)

	var word string
	word = "abc"
	u := context{"fgh", "de"}

	pAddZero, _ := hpylm.CalcProb(word, u, float64(base))
	if !(pAddZero == float64(base)) {
		t.Error("pAddZero = ", pAddZero, "base = ", base)
	}

	body := (1.0 - d*1.0) / (theta + 1.0)
	smoothing := (theta + d*1.0) / (theta + 1.0)
	pCorrect := body + smoothing*(body+smoothing*(body+smoothing*base))

	hpylm.AddCustomer(word, u, float64(base), hpylm.addCustomerBaseNull)
	pAddOne, probsAddOne := hpylm.CalcProb(word, u, float64(base))
	fmt.Println(pAddOne, probsAddOne)
	if !(pAddOne == float64(pCorrect)) {
		t.Error("pAddOne = ", pAddOne, "pCorrect = ", pCorrect)
	}
	if !(pAddOne >= pAddZero) {
		t.Error("pAddOne = ", pAddOne, "pAddZero = ", pAddZero)
	}
	customerCountRestOFfghde := hpylm.restaurants["fgh<concat>de"].customerCount["abc"]
	totalCustomerCountRestOFfghde := hpylm.restaurants["fgh<concat>de"].totalCustomerCount
	totalTableCountForCustomerRestOFfghde := hpylm.restaurants["fgh<concat>de"].totalTableCountForCustomer["abc"]
	totalTableCountRestOFfghde := hpylm.restaurants["fgh<concat>de"].totalTableCount
	customerCountRestOFde := hpylm.restaurants["de"].customerCount["abc"]
	totalCustomerCountRestOFde := hpylm.restaurants["de"].totalCustomerCount
	totalTableCountForCustomerRestOFde := hpylm.restaurants["de"].totalTableCountForCustomer["abc"]
	totalTableCountRestOFde := hpylm.restaurants["de"].totalTableCount
	customerCountRestOF := hpylm.restaurants[""].customerCount["abc"]
	totalCustomerCountRestOF := hpylm.restaurants[""].totalCustomerCount
	totalTableCountForCustomerRestOF := hpylm.restaurants[""].totalTableCountForCustomer["abc"]
	totalTableCountRestOF := hpylm.restaurants[""].totalTableCount
	addOnePrams := make([]newUint, 0, 0)
	addOnePrams = append(addOnePrams, newUint(customerCountRestOFfghde))
	addOnePrams = append(addOnePrams, newUint(totalCustomerCountRestOFfghde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountForCustomerRestOFfghde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountRestOFfghde))
	addOnePrams = append(addOnePrams, newUint(customerCountRestOFde))
	addOnePrams = append(addOnePrams, newUint(totalCustomerCountRestOFde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountForCustomerRestOFde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountRestOFde))
	addOnePrams = append(addOnePrams, newUint(customerCountRestOF))
	addOnePrams = append(addOnePrams, newUint(totalCustomerCountRestOF))
	addOnePrams = append(addOnePrams, newUint(totalTableCountForCustomerRestOF))
	addOnePrams = append(addOnePrams, newUint(totalTableCountRestOF))

	for i := 0; i < epoch; i++ {
		hpylm.AddCustomer(word, u, float64(base), hpylm.addCustomerBaseNull)
	}
	pAddMany, probsAddMany := hpylm.CalcProb(word, u, float64(base))
	if !(pAddMany >= pAddOne) {
		t.Error("pAddMany = ", pAddMany, "pAddOne = ", pAddOne)
	}
	for i := 0; i < len(probsAddMany); i++ {
		if !(probsAddMany[i] >= probsAddOne[i]) {
			t.Error("probsAddMany[i] = ", probsAddMany[i], "probsAddOne[i] = ", probsAddOne[i], "i = ", i)
		}
	}
	customerCountRestOFfghde = hpylm.restaurants["fgh<concat>de"].customerCount["abc"]
	totalCustomerCountRestOFfghde = hpylm.restaurants["fgh<concat>de"].totalCustomerCount
	totalTableCountForCustomerRestOFfghde = hpylm.restaurants["fgh<concat>de"].totalTableCountForCustomer["abc"]
	totalTableCountRestOFfghde = hpylm.restaurants["fgh<concat>de"].totalTableCount
	customerCountRestOFde = hpylm.restaurants["de"].customerCount["abc"]
	totalCustomerCountRestOFde = hpylm.restaurants["de"].totalCustomerCount
	totalTableCountForCustomerRestOFde = hpylm.restaurants["de"].totalTableCountForCustomer["abc"]
	totalTableCountRestOFde = hpylm.restaurants["de"].totalTableCount
	customerCountRestOF = hpylm.restaurants[""].customerCount["abc"]
	totalCustomerCountRestOF = hpylm.restaurants[""].totalCustomerCount
	totalTableCountForCustomerRestOF = hpylm.restaurants[""].totalTableCountForCustomer["abc"]
	totalTableCountRestOF = hpylm.restaurants[""].totalTableCount
	addManyPrams := make([]newUint, 0, 0)
	addManyPrams = append(addManyPrams, newUint(customerCountRestOFfghde))
	addManyPrams = append(addManyPrams, newUint(totalCustomerCountRestOFfghde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountForCustomerRestOFfghde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountRestOFfghde))
	addManyPrams = append(addManyPrams, newUint(customerCountRestOFde))
	addManyPrams = append(addManyPrams, newUint(totalCustomerCountRestOFde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountForCustomerRestOFde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountRestOFde))
	addManyPrams = append(addManyPrams, newUint(customerCountRestOF))
	addManyPrams = append(addManyPrams, newUint(totalCustomerCountRestOF))
	addManyPrams = append(addManyPrams, newUint(totalTableCountForCustomerRestOF))
	addManyPrams = append(addManyPrams, newUint(totalTableCountRestOF))
	for i := 0; i < len(addOnePrams); i++ {
		if !(addManyPrams[i] >= addOnePrams[i]) {
			t.Error("addManyPrams[i] = ", addManyPrams[i], "addOnePrams[i] = ", addOnePrams[i], "i = ", i)
		}
	}

	for i := 0; i < epoch; i++ {
		hpylm.RemoveCustomer(word, u, hpylm.removeCustomerBaseNull)
	}
	pRemoveMany, probsRemoveMany := hpylm.CalcProb(word, u, float64(base))
	if !(pRemoveMany == pAddOne) {
		t.Error("pRemoveMany = ", pRemoveMany, "pAddOne = ", pAddOne)
	}
	for i := 0; i < len(probsRemoveMany); i++ {
		if !(probsRemoveMany[i] == probsAddOne[i]) {
			t.Error("probsRemoveMany[i] = ", probsRemoveMany[i], "probsAddOne[i] = ", probsAddOne[i], "i = ", i)
		}
	}

	hpylm.RemoveCustomer(word, u, hpylm.removeCustomerBaseNull)
	pRemoveOne, _ := hpylm.CalcProb(word, u, float64(base))
	if !(pRemoveOne == pAddZero) {
		t.Error("pRemoveOne = ", pRemoveOne, "pAddZero = ", pAddZero)
	}

	if !(len(hpylm.restaurants) == 0) {
		t.Error("hpylm.restaurants = ", hpylm.restaurants)
	}
}

func TestPerformanceOfHPYLM(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	var theta float64
	var d float64
	var base float64
	var epoch int
	var maxN int
	maxN = 3
	base = float64(1.0 / 2097152.0) // 1 / 2^21 , size of character vocabulary in utf-8 encodeing
	theta = 1.0
	d = 0.1
	epoch = 10
	var hpylm LmModel
	hpylm = NewHPYLM(maxN-1, theta, d, 1.0, 1.0, 1.0, 1.0, base)

	var interporationRates []float64
	interporationRates = []float64{0.1, 0.1, 0.1}
	var interporatedNgram LmModel
	interporatedNgram = NewNgram(maxN, interporationRates, base)

	dataContainerForTrain := NewDataContainerFromAnnotatedData("../alice.train.txt")
	dataContainerForTest := NewDataContainerFromAnnotatedData("../alice.test.txt")
	for e := 0; e < epoch; e++ {
		hpylm.Train(dataContainerForTrain)
	}
	interporatedNgram.Train(dataContainerForTrain)

	perplexityOfHpylm := CalcPerplexity(hpylm, dataContainerForTest)
	perplexityOfInterporatedNgram := CalcPerplexity(interporatedNgram, dataContainerForTest)
	if !(perplexityOfHpylm < perplexityOfInterporatedNgram) {
		t.Error("probably error! a perplexity of HPYLM is expected to be lower than a perplexity of interporated n-gram. ", "perplexityOfHpylm = ", perplexityOfHpylm, "perplexityOfInterporatedNgram = ", perplexityOfInterporatedNgram)
	}
}
