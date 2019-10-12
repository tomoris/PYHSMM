package NPYLM

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

	pAddZero, _ := hpylm.CalcProb(word, u, newFloat(base))
	if !(pAddZero == newFloat(base)) {
		t.Error("pAddZero = ", pAddZero, "base = ", base)
	}

	body := (1.0 - d*1.0) / (theta + 1.0)
	smoothing := (theta + d*1.0) / (theta + 1.0)
	pCorrect := body + smoothing*(body+smoothing*(body+smoothing*base))

	hpylm.AddCustomer(word, u, newFloat(base), hpylm.addCustomerBaseNull)
	pAddOne, probsAddOne := hpylm.CalcProb(word, u, newFloat(base))
	fmt.Println(pAddOne, probsAddOne)
	if !(pAddOne == newFloat(pCorrect)) {
		t.Error("pAddOne = ", pAddOne, "pCorrect = ", pCorrect)
	}
	if !(pAddOne >= pAddZero) {
		t.Error("pAddOne = ", pAddOne, "pAddZero = ", pAddZero)
	}
	customerCount_rest_fghde := hpylm.restaurants["fgh<concat>de"].customerCount["abc"]
	totalCustomerCount_rest_fghde := hpylm.restaurants["fgh<concat>de"].totalCustomerCount
	totalTableCountForCustomer_rest_fghde := hpylm.restaurants["fgh<concat>de"].totalTableCountForCustomer["abc"]
	totalTableCount_rest_fghde := hpylm.restaurants["fgh<concat>de"].totalTableCount
	customerCount_rest_de := hpylm.restaurants["de"].customerCount["abc"]
	totalCustomerCount_rest_de := hpylm.restaurants["de"].totalCustomerCount
	totalTableCountForCustomer_rest_de := hpylm.restaurants["de"].totalTableCountForCustomer["abc"]
	totalTableCount_rest_de := hpylm.restaurants["de"].totalTableCount
	customerCount_rest_ := hpylm.restaurants[""].customerCount["abc"]
	totalCustomerCount_rest_ := hpylm.restaurants[""].totalCustomerCount
	totalTableCountForCustomer_rest_ := hpylm.restaurants[""].totalTableCountForCustomer["abc"]
	totalTableCount_rest_ := hpylm.restaurants[""].totalTableCount
	addOne_prams := make([]newUint, 0, 0)
	addOne_prams = append(addOne_prams, newUint(customerCount_rest_fghde))
	addOne_prams = append(addOne_prams, newUint(totalCustomerCount_rest_fghde))
	addOne_prams = append(addOne_prams, newUint(totalTableCountForCustomer_rest_fghde))
	addOne_prams = append(addOne_prams, newUint(totalTableCount_rest_fghde))
	addOne_prams = append(addOne_prams, newUint(customerCount_rest_de))
	addOne_prams = append(addOne_prams, newUint(totalCustomerCount_rest_de))
	addOne_prams = append(addOne_prams, newUint(totalTableCountForCustomer_rest_de))
	addOne_prams = append(addOne_prams, newUint(totalTableCount_rest_de))
	addOne_prams = append(addOne_prams, newUint(customerCount_rest_))
	addOne_prams = append(addOne_prams, newUint(totalCustomerCount_rest_))
	addOne_prams = append(addOne_prams, newUint(totalTableCountForCustomer_rest_))
	addOne_prams = append(addOne_prams, newUint(totalTableCount_rest_))

	for i := 0; i < epoch; i++ {
		hpylm.AddCustomer(word, u, newFloat(base), hpylm.addCustomerBaseNull)
	}
	pAddMany, probsAddMany := hpylm.CalcProb(word, u, newFloat(base))
	if !(pAddMany >= pAddOne) {
		t.Error("pAddMany = ", pAddMany, "pAddOne = ", pAddOne)
	}
	for i := 0; i < len(probsAddMany); i++ {
		if !(probsAddMany[i] >= probsAddOne[i]) {
			t.Error("probsAddMany[i] = ", probsAddMany[i], "probsAddOne[i] = ", probsAddOne[i], "i = ", i)
		}
	}
	customerCount_rest_fghde = hpylm.restaurants["fgh<concat>de"].customerCount["abc"]
	totalCustomerCount_rest_fghde = hpylm.restaurants["fgh<concat>de"].totalCustomerCount
	totalTableCountForCustomer_rest_fghde = hpylm.restaurants["fgh<concat>de"].totalTableCountForCustomer["abc"]
	totalTableCount_rest_fghde = hpylm.restaurants["fgh<concat>de"].totalTableCount
	customerCount_rest_de = hpylm.restaurants["de"].customerCount["abc"]
	totalCustomerCount_rest_de = hpylm.restaurants["de"].totalCustomerCount
	totalTableCountForCustomer_rest_de = hpylm.restaurants["de"].totalTableCountForCustomer["abc"]
	totalTableCount_rest_de = hpylm.restaurants["de"].totalTableCount
	customerCount_rest_ = hpylm.restaurants[""].customerCount["abc"]
	totalCustomerCount_rest_ = hpylm.restaurants[""].totalCustomerCount
	totalTableCountForCustomer_rest_ = hpylm.restaurants[""].totalTableCountForCustomer["abc"]
	totalTableCount_rest_ = hpylm.restaurants[""].totalTableCount
	addMany_prams := make([]newUint, 0, 0)
	addMany_prams = append(addMany_prams, newUint(customerCount_rest_fghde))
	addMany_prams = append(addMany_prams, newUint(totalCustomerCount_rest_fghde))
	addMany_prams = append(addMany_prams, newUint(totalTableCountForCustomer_rest_fghde))
	addMany_prams = append(addMany_prams, newUint(totalTableCount_rest_fghde))
	addMany_prams = append(addMany_prams, newUint(customerCount_rest_de))
	addMany_prams = append(addMany_prams, newUint(totalCustomerCount_rest_de))
	addMany_prams = append(addMany_prams, newUint(totalTableCountForCustomer_rest_de))
	addMany_prams = append(addMany_prams, newUint(totalTableCount_rest_de))
	addMany_prams = append(addMany_prams, newUint(customerCount_rest_))
	addMany_prams = append(addMany_prams, newUint(totalCustomerCount_rest_))
	addMany_prams = append(addMany_prams, newUint(totalTableCountForCustomer_rest_))
	addMany_prams = append(addMany_prams, newUint(totalTableCount_rest_))
	for i := 0; i < len(addOne_prams); i++ {
		if !(addMany_prams[i] >= addOne_prams[i]) {
			t.Error("addMany_prams[i] = ", addMany_prams[i], "addOne_prams[i] = ", addOne_prams[i], "i = ", i)
		}
	}

	for i := 0; i < epoch; i++ {
		hpylm.RemoveCustomer(word, u, hpylm.removeCustomerBaseNull)
	}
	pRemoveMany, probsRemoveMany := hpylm.CalcProb(word, u, newFloat(base))
	if !(pRemoveMany == pAddOne) {
		t.Error("pRemoveMany = ", pRemoveMany, "pAddOne = ", pAddOne)
	}
	for i := 0; i < len(probsRemoveMany); i++ {
		if !(probsRemoveMany[i] == probsAddOne[i]) {
			t.Error("probsRemoveMany[i] = ", probsRemoveMany[i], "probsAddOne[i] = ", probsAddOne[i], "i = ", i)
		}
	}

	hpylm.RemoveCustomer(word, u, hpylm.removeCustomerBaseNull)
	pRemoveOne, _ := hpylm.CalcProb(word, u, newFloat(base))
	if !(pRemoveOne == pAddZero) {
		t.Error("pRemoveOne = ", pRemoveOne, "pAddZero = ", pAddZero)
	}

	if !(len(hpylm.restaurants) == 0) {
		t.Error("hpylm.restaurants = ", hpylm.restaurants)
	}
}
