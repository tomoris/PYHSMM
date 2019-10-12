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
	customerCountRestISfghde := hpylm.restaurants["fgh<concat>de"].customerCount["abc"]
	totalCustomerCountRestISfghde := hpylm.restaurants["fgh<concat>de"].totalCustomerCount
	totalTableCountForCustomerRestISfghde := hpylm.restaurants["fgh<concat>de"].totalTableCountForCustomer["abc"]
	totalTableCountRestISfghde := hpylm.restaurants["fgh<concat>de"].totalTableCount
	customerCountRestISde := hpylm.restaurants["de"].customerCount["abc"]
	totalCustomerCountRestISde := hpylm.restaurants["de"].totalCustomerCount
	totalTableCountForCustomerRestISde := hpylm.restaurants["de"].totalTableCountForCustomer["abc"]
	totalTableCountRestISde := hpylm.restaurants["de"].totalTableCount
	customerCountRestIS := hpylm.restaurants[""].customerCount["abc"]
	totalCustomerCountRestIS := hpylm.restaurants[""].totalCustomerCount
	totalTableCountForCustomerRestIS := hpylm.restaurants[""].totalTableCountForCustomer["abc"]
	totalTableCountRestIS := hpylm.restaurants[""].totalTableCount
	addOnePrams := make([]newUint, 0, 0)
	addOnePrams = append(addOnePrams, newUint(customerCountRestISfghde))
	addOnePrams = append(addOnePrams, newUint(totalCustomerCountRestISfghde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountForCustomerRestISfghde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountRestISfghde))
	addOnePrams = append(addOnePrams, newUint(customerCountRestISde))
	addOnePrams = append(addOnePrams, newUint(totalCustomerCountRestISde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountForCustomerRestISde))
	addOnePrams = append(addOnePrams, newUint(totalTableCountRestISde))
	addOnePrams = append(addOnePrams, newUint(customerCountRestIS))
	addOnePrams = append(addOnePrams, newUint(totalCustomerCountRestIS))
	addOnePrams = append(addOnePrams, newUint(totalTableCountForCustomerRestIS))
	addOnePrams = append(addOnePrams, newUint(totalTableCountRestIS))

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
	customerCountRestISfghde = hpylm.restaurants["fgh<concat>de"].customerCount["abc"]
	totalCustomerCountRestISfghde = hpylm.restaurants["fgh<concat>de"].totalCustomerCount
	totalTableCountForCustomerRestISfghde = hpylm.restaurants["fgh<concat>de"].totalTableCountForCustomer["abc"]
	totalTableCountRestISfghde = hpylm.restaurants["fgh<concat>de"].totalTableCount
	customerCountRestISde = hpylm.restaurants["de"].customerCount["abc"]
	totalCustomerCountRestISde = hpylm.restaurants["de"].totalCustomerCount
	totalTableCountForCustomerRestISde = hpylm.restaurants["de"].totalTableCountForCustomer["abc"]
	totalTableCountRestISde = hpylm.restaurants["de"].totalTableCount
	customerCountRestIS = hpylm.restaurants[""].customerCount["abc"]
	totalCustomerCountRestIS = hpylm.restaurants[""].totalCustomerCount
	totalTableCountForCustomerRestIS = hpylm.restaurants[""].totalTableCountForCustomer["abc"]
	totalTableCountRestIS = hpylm.restaurants[""].totalTableCount
	addManyPrams := make([]newUint, 0, 0)
	addManyPrams = append(addManyPrams, newUint(customerCountRestISfghde))
	addManyPrams = append(addManyPrams, newUint(totalCustomerCountRestISfghde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountForCustomerRestISfghde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountRestISfghde))
	addManyPrams = append(addManyPrams, newUint(customerCountRestISde))
	addManyPrams = append(addManyPrams, newUint(totalCustomerCountRestISde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountForCustomerRestISde))
	addManyPrams = append(addManyPrams, newUint(totalTableCountRestISde))
	addManyPrams = append(addManyPrams, newUint(customerCountRestIS))
	addManyPrams = append(addManyPrams, newUint(totalCustomerCountRestIS))
	addManyPrams = append(addManyPrams, newUint(totalTableCountForCustomerRestIS))
	addManyPrams = append(addManyPrams, newUint(totalTableCountRestIS))
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
