package bayselm

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/cheggaaa/pb/v3"
	"gonum.org/v1/gonum/stat/distuv"
)

type table newUint // number of customer (input word) in the table

type restaurant struct {
	tables map[string][]table // word to tables
	// simple p(word|context) = customerCount[word] / totalCustomerCount
	customerCount              map[string]newUint // word to count int the restaurant.
	totalCustomerCount         newUint
	totalTableCountForCustomer map[string]newUint // word to number of word serving tables
	totalTableCount            newUint            // number of tables

	stop newUint // number of stop for stop probability in n-gram
	pass newUint // number of pass for stop probability in n-gram
}

func (rst *restaurant) save() ([]byte, *restaurantJSON) {
	rstJSON := &restaurantJSON{
		Tables:                     rst.tables,
		CustomerCount:              rst.customerCount,
		TotalCustomerCount:         rst.totalCustomerCount,
		TotalTableCountForCustomer: rst.totalTableCountForCustomer,
		TotalTableCount:            rst.totalTableCount,

		Stop: rst.stop,
		Pass: rst.pass,
	}
	v, err := json.Marshal(&rstJSON)
	if err != nil {
		panic("save error in PYHSMM")
	}
	return v, rstJSON
}

func (rst *restaurant) load(v []byte) {
	rstJSON := new(restaurantJSON)

	err := json.Unmarshal(v, &rstJSON)
	if err != nil {
		panic("load error in restaurant")
	}

	rst.tables = rstJSON.Tables
	rst.customerCount = rstJSON.CustomerCount
	rst.totalCustomerCount = rstJSON.TotalCustomerCount
	rst.totalTableCountForCustomer = rstJSON.TotalTableCountForCustomer
	rst.totalTableCount = rstJSON.TotalTableCount

	rst.stop = rstJSON.Stop
	rst.pass = rstJSON.Pass
	return
}

// HPYLM contains n-gram parameters as restaurants.
type HPYLM struct {
	restaurants map[string]*restaurant // context to restaurant

	maxDepth int
	theta    []float64 // parameters for Pitman-Yor process
	d        []float64 // parameters for Pitman-Yor process
	gammaA   []float64 // hyper-parameters using gamma distributionfor to estimate theta
	gammaB   []float64 // hyper-parameters using gamma distributionfor to estimate theta
	betaA    []float64 // hyper-parameters using beta distributionfor to estimate d
	betaB    []float64 // hyper-parameters using beta distributionfor to estimate d
	Base     float64
}

func newRestaurant() *restaurant {
	rst := new(restaurant)
	rst.tables = make(map[string][]table)
	rst.customerCount = make(map[string]newUint)
	rst.totalCustomerCount = 0
	rst.totalTableCountForCustomer = make(map[string]newUint)
	rst.totalTableCount = 0
	return rst
}

func (rst *restaurant) addCustomer(word string, u context, k newUint) bool {
	addTbl := false

	if rst.totalTableCountForCustomer[word] > k {
		rst.tables[word][k]++
	} else {
		// add new table
		rst.tables[word] = append(rst.tables[word], 1)
		rst.totalTableCountForCustomer[word]++
		rst.totalTableCount++
		addTbl = true
	}
	rst.customerCount[word]++
	rst.totalCustomerCount++
	return addTbl
}

func (rst *restaurant) removeCustomer(word string, u context, k newUint) (bool, bool) {
	removeTbl := false
	removeRst := false

	rst.tables[word][k]--
	rst.customerCount[word]--
	rst.totalCustomerCount--
	if rst.tables[word][k] == 0 {
		// remove k-th table
		rst.tables[word] = append(rst.tables[word][:k], rst.tables[word][k+1:]...)
		rst.totalTableCountForCustomer[word]--
		rst.totalTableCount--
		removeTbl = true
		if rst.totalTableCount == 0 {
			removeRst = true
		}
	}
	if len(rst.tables[word]) == 0 {
		delete(rst.tables, word)
	}
	return removeTbl, removeRst
}

// NewHPYLM returns HPYLM instance.
func NewHPYLM(maxDepth int, initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, Base float64) *HPYLM {
	hpylm := new(HPYLM)
	hpylm.restaurants = make(map[string]*restaurant)
	hpylm.maxDepth = maxDepth
	hpylm.theta = make([]float64, hpylm.maxDepth+1, hpylm.maxDepth+1)
	hpylm.d = make([]float64, hpylm.maxDepth+1, hpylm.maxDepth+1)
	hpylm.gammaA = make([]float64, hpylm.maxDepth+1, hpylm.maxDepth+1)
	hpylm.gammaB = make([]float64, hpylm.maxDepth+1, hpylm.maxDepth+1)
	hpylm.betaA = make([]float64, hpylm.maxDepth+1, hpylm.maxDepth+1)
	hpylm.betaB = make([]float64, hpylm.maxDepth+1, hpylm.maxDepth+1)
	for i := 0; i <= hpylm.maxDepth; i++ {
		hpylm.theta[i] = float64(initialTheta)
		hpylm.d[i] = float64(initialD)
		hpylm.gammaA[i] = float64(gammaA)
		hpylm.gammaB[i] = float64(gammaB)
		hpylm.betaA[i] = float64(betaA)
		hpylm.betaB[i] = float64(betaB)
	}
	if maxDepth <= 0 {
		panic("range of maxDepth is 0 to 255")
	}
	if initialD <= 0.0 || initialD >= 1.0 {
		panic("range of initialD is 0.0 to 1.0")
	}
	if initialTheta <= 0.0 {
		panic("range of initialTheta is range 0.0 to inf")
	}
	if Base <= 0.0 || Base >= 1.0 {
		panic("range of Base is 0.0 to 1.0")
	}

	hpylm.Base = float64(Base)
	return hpylm
}

// AddCustomer adds n-gram parameters.
func (hpylm *HPYLM) AddCustomer(word string, u context, base float64, addBaseFunc func(string)) {
	_, probs := hpylm.CalcProb(word, u, base)
	hpylm.addCustomerRecursively(word, u, probs, base, addBaseFunc)
	return
}

func (hpylm *HPYLM) addStopAndPassCount(u context) {
	rst, ok := hpylm.restaurants[strings.Join(u, concat)]
	if !ok {
		errMsg := fmt.Sprintf("addStopAndPassCount error. context u (%v) does not exist", u)
		panic(errMsg)
	}
	rst.stop++
	for i := 1; i <= len(u); i++ {
		rst, ok := hpylm.restaurants[strings.Join(u[i:], concat)]
		if !ok {
			errMsg := fmt.Sprintf("addStopAndPassCount error. context u (%v) does not exist", u[i:])
			panic(errMsg)
		}
		rst.pass++
	}
	return
}

// rewrite this later to a fuction for add customer in VPYLM  as character level LM's parameters
func (hpylm *HPYLM) addCustomerBaseNull(word string) {
	return
}

func (hpylm *HPYLM) addCustomerRecursively(word string, u context, probs []float64, base float64, addBaseFunc func(string)) {
	theta := hpylm.theta[len(u)]
	d := hpylm.d[len(u)]
	rst, ok := hpylm.restaurants[strings.Join(u, concat)]
	if !ok {
		hpylm.restaurants[strings.Join(u, concat)] = newRestaurant()
		rst = hpylm.restaurants[strings.Join(u, concat)]
	}
	tbls, ok := rst.tables[word]
	if !ok {
		rst.tables[word] = []table{}
		tbls = rst.tables[word]
	}
	tableNum := rst.totalTableCountForCustomer[word]
	score := float64(0.0)
	scoreArray := make([]float64, tableNum+1, tableNum+1)
	sumScore := float64(0.0)
	for k, tbl := range tbls {
		// score = float64(tbl) - (d * float64(rst.totalTableCountForCustomer[word]))
		score = float64(tbl) - d
		scoreArray[k] = math.Max(0.0, float64(score))
		sumScore += scoreArray[k]
	}
	smoothingCoefficient := (theta + (d * float64(rst.totalTableCount))) / (theta + float64(rst.totalCustomerCount))
	if len(u) == 0 {
		scoreArray[tableNum] = smoothingCoefficient*base + math.SmallestNonzeroFloat64
	} else {
		scoreArray[tableNum] = smoothingCoefficient*probs[len(u)-1] + math.SmallestNonzeroFloat64 // hpylm.CalcProb(word, u[1:])
	}
	sumScore += scoreArray[tableNum]

	// sampling
	r := rand.Float64()*sumScore - math.SmallestNonzeroFloat64
	sumScore = 0.0
	k := newUint(0)
	for {
		sumScore += scoreArray[k]
		if sumScore > r {
			break
		}
		k++
		if k >= tableNum+1 {
			panic("sampling error in HPYLM")
		}
	}

	// add and recursive
	addTbl := rst.addCustomer(word, u, k)
	if addTbl {
		if len(u) > 0 {
			hpylm.addCustomerRecursively(word, u[1:], probs, base, addBaseFunc)
		} else {
			addBaseFunc(word)
			// hpylm.addCustomerBase(word)
		}
	}
	return
}

// rewrite this later to a fuction for remove customer in VPYLM  as character level LM's parameters
func (hpylm *HPYLM) removeCustomerBaseNull(word string) {
	return
}

// RemoveCustomer removes n-gram parameters.
func (hpylm *HPYLM) RemoveCustomer(word string, u context, removeBaseFunc func(string)) {
	rst, ok := hpylm.restaurants[strings.Join(u, concat)]
	if !ok {
		errMsg := fmt.Sprintf("remove error. context u (%v) does not exist in HPYLM", u)
		panic(errMsg)
	}
	tbls, ok := rst.tables[word]
	if !ok {
		errMsg := fmt.Sprintf("remove error. word (%v) does not exist in restaurant u (%v) HPYLM", word, u)
		panic(errMsg)
	}
	tableNum := rst.totalTableCountForCustomer[word]
	score := float64(0.0)
	scoreArray := make([]float64, tableNum, tableNum)
	sumScore := float64(0.0)
	for k, tbl := range tbls {
		score = float64(tbl)
		scoreArray[k] = score
		sumScore += scoreArray[k]
	}

	// sampling
	r := rand.Float64() * sumScore
	sumScore = 0.0
	k := newUint(0)
	for {
		sumScore += scoreArray[k]
		if sumScore > r {
			break
		}
		k++
		if k >= tableNum {
			panic("sampling error in HPYLM")
		}
	}

	// remove and recursive
	removeTbl, removeRst := rst.removeCustomer(word, u, k)
	if removeTbl {
		if len(u) > 0 {
			hpylm.RemoveCustomer(word, u[1:], removeBaseFunc)
		} else {
			removeBaseFunc(word)
			// hpylm.removeCustomerBase(word)
		}
	}
	if removeRst {
		delete(hpylm.restaurants, strings.Join(u, concat))
	}

	return
}

func (hpylm *HPYLM) removeStopAndPassCount(word string, u context) {
	rst, ok := hpylm.restaurants[strings.Join(u, concat)]
	if !ok {
		errMsg := fmt.Sprintf("removeStopAndPassCount error. context u (%v) does not exist", u)
		panic(errMsg)
	}
	if rst.stop == 0 {
		errMsg := fmt.Sprintf("removeStopAndPassCount error. rst.stop of context u (%v) == 0", u)
		panic(errMsg)
	}
	rst.stop--

	for i := 1; i <= len(u); i++ {
		rst, ok := hpylm.restaurants[strings.Join(u[i:], concat)]
		if !ok {
			errMsg := fmt.Sprintf("removeStopAndPassCount error. context u (%v) does not exist", u[i:])
			panic(errMsg)
		}
		if rst.pass == 0 {
			errMsg := fmt.Sprintf("removeStopAndPassCount error. rst.stop of context u (%v) == 0", u)
			panic(errMsg)
		}
		rst.pass--
	}
	return
}

// CalcProb returns n-gram prrobability.
func (hpylm *HPYLM) CalcProb(word string, u context, base float64) (float64, []float64) {
	if len(u) > hpylm.maxDepth {
		panic("maximum depth error")
	}

	probBodies := make([]float64, len(u)+1, len(u)+1)
	smoothingCoefficients := make([]float64, len(u)+1, len(u)+1)
	hpylm.calcProbRecursively(word, u, probBodies, smoothingCoefficients)
	probs := make([]float64, len(u)+1, len(u)+1)
	p := base
	for i, pTmp := range probBodies {
		p = pTmp + (smoothingCoefficients[i] * p)
		probs[i] = p
	}

	return p + math.SmallestNonzeroFloat64, probs
}

func (hpylm *HPYLM) calcProbRecursively(word string, u context, probBodies []float64, smoothingCoefficients []float64) {
	theta := hpylm.theta[len(u)]
	d := hpylm.d[len(u)]
	rst, ok := hpylm.restaurants[strings.Join(u, concat)]
	pTmp := float64(0.0)
	smoothingCoefficient := float64(1.0)
	if ok {
		pTmp = (float64(rst.customerCount[word]) - (d * float64(rst.totalTableCountForCustomer[word]))) / (theta + float64(rst.totalCustomerCount))
		smoothingCoefficient = (theta + (d * float64(rst.totalTableCount))) / (theta + float64(rst.totalCustomerCount))
	}

	probBodies[len(u)] = pTmp
	smoothingCoefficients[len(u)] = smoothingCoefficient

	if len(u) != 0 {
		hpylm.calcProbRecursively(word, u[1:], probBodies, smoothingCoefficients)
	}
	return
}

func (hpylm *HPYLM) estimateHyperPrameters() {
	uSliceEachN := make([][]context, hpylm.maxDepth+1, hpylm.maxDepth+1)
	for n := 0; n < hpylm.maxDepth+1; n++ {
		uSlice := make([]context, 0, 0)
		uSliceEachN[n] = uSlice
	}
	for uStr := range hpylm.restaurants {
		if uStr == "" {
			uSliceEachN[0] = append(uSliceEachN[0], context{""})
			continue
		}
		u := context(strings.Split(uStr, concat))
		uSliceEachN[len(u)] = append(uSliceEachN[len(u)], u)
	}

	for n := 0; n < hpylm.maxDepth+1; n++ {
		aForTheta := float64(hpylm.gammaA[n])
		bForTheta := float64(hpylm.gammaB[n])
		aForD := float64(hpylm.betaA[n])
		bForD := float64(hpylm.betaB[n])
		for _, u := range uSliceEachN[n] {
			totalTableCount := int(hpylm.restaurants[strings.Join(u, concat)].totalTableCount)
			if totalTableCount < 2 {
				continue
			}
			betaDist := distuv.Beta{}
			thetaTmp := float64(hpylm.theta[n])
			dTmp := float64(hpylm.d[n])
			betaDist.Alpha = thetaTmp + 1.0
			betaDist.Beta = float64(hpylm.restaurants[strings.Join(u, concat)].totalCustomerCount) - 1.0
			xu := betaDist.Rand()
			for t := 1; t < totalTableCount; t++ {
				bernoulliDist := distuv.Bernoulli{}
				bernoulliDist.P = thetaTmp / (thetaTmp + (dTmp * float64(t)))
				y := bernoulliDist.Rand()

				aForTheta += y
				bForTheta -= math.Log(xu)
				aForD += (1.0 - y)
			}
			for _, tbls := range hpylm.restaurants[strings.Join(u, concat)].tables {
				for _, customerCount := range tbls {
					if int(customerCount) < 2 {
						continue
					}
					for j := 1; j < int(customerCount); j++ {
						bernoulliDist := distuv.Bernoulli{}
						bernoulliDist.P = (float64(j) - 1.0) / (float64(j) - dTmp)
						z := bernoulliDist.Rand()
						bForD += (1.0 - z)
					}
				}
			}
		}
		gammaDist := distuv.Gamma{}
		gammaDist.Alpha = aForTheta
		gammaDist.Beta = bForTheta
		betaDist := distuv.Beta{}
		betaDist.Alpha = aForD
		betaDist.Beta = bForD
		hpylm.theta[n] = float64(gammaDist.Rand())
		hpylm.d[n] = float64(betaDist.Rand())
		if hpylm.theta[n] < 0.0 {
			panic("theta estimation error")
		}
		if hpylm.d[n] < 0.0 {
			panic("d estimation error")
		}
	}
	return
}

// Train train n-gram parameters from given word sequences.
func (hpylm *HPYLM) Train(dataContainer *DataContainer) {
	removeFlag := true
	if len(hpylm.restaurants) == 0 { // epoch == 0
		removeFlag = false
	}
	bar := pb.StartNew(dataContainer.Size)
	randIndexes := rand.Perm(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i++ {
		bar.Add(1)
		r := randIndexes[i]
		wordSeq := dataContainer.SamplingWordSeqs[r]
		if removeFlag {
			u := make(context, 0, hpylm.maxDepth)
			for n := 0; n < hpylm.maxDepth; n++ {
				u = append(u, bos)
			}
			for _, word := range wordSeq {
				hpylm.RemoveCustomer(word, u, hpylm.removeCustomerBaseNull)
				u = append(u[1:], word)
			}
		}
		u := make(context, 0, hpylm.maxDepth)
		for n := 0; n < hpylm.maxDepth; n++ {
			u = append(u, bos)
		}
		for _, word := range wordSeq {
			hpylm.AddCustomer(word, u, hpylm.Base, hpylm.addCustomerBaseNull)
			u = append(u[1:], word)
		}
	}
	bar.Finish()
	hpylm.estimateHyperPrameters()
	return
}

// ReturnNgramProb returns n-gram probability.
// This is used for interface of LmModel.
func (hpylm *HPYLM) ReturnNgramProb(word string, u context) float64 {
	p, _ := hpylm.CalcProb(word, u, hpylm.Base)
	return p
}

// ReturnMaxN returns maximum length of n-gram.
// This is used for interface of LmModel.
func (hpylm *HPYLM) ReturnMaxN() int {
	return hpylm.maxDepth + 1
}

// Save returns json.Marshal(hpylmJSON) and hpylmJSON.
// hpylmJSON is struct to save. its variables can be exported.
func (hpylm *HPYLM) save() ([]byte, interface{}) {
	hpylmJSON := &hPYLMJSON{
		Restaurants: func(rsts map[string]*restaurant) map[string]*restaurantJSON {
			rstsJSON := make(map[string]*restaurantJSON)
			for key, rst := range rsts {
				_, rstJSON := rst.save()
				rstsJSON[key] = rstJSON
			}
			return rstsJSON
		}(hpylm.restaurants),

		MaxDepth: hpylm.maxDepth,
		Theta:    hpylm.theta,
		D:        hpylm.d,
		GammaA:   hpylm.gammaA,
		GammaB:   hpylm.gammaB,
		BetaA:    hpylm.betaA,
		BetaB:    hpylm.betaB,
		Base:     hpylm.Base,
	}
	v, err := json.Marshal(&hpylmJSON)
	if err != nil {
		panic("save error in HPYLM")
	}
	return v, hpylmJSON
}

// Load hpylm.
func (hpylm *HPYLM) load(v []byte) {
	hpylmJSON := new(hPYLMJSON)
	err := json.Unmarshal(v, &hpylmJSON)
	if err != nil {
		panic("load error in HPYLM")
	}
	hpylm.restaurants = func(hpylmJSON *hPYLMJSON) map[string]*restaurant {
		rsts := make(map[string]*restaurant)
		for key, rstJSON := range hpylmJSON.Restaurants {
			rstV, err := json.Marshal(&rstJSON)
			if err != nil {
				panic("load error in load restaurants in HPYLM")
			}
			rst := newRestaurant()
			rst.load(rstV)
			rsts[key] = rst
		}
		return rsts
	}(hpylmJSON)

	hpylm.maxDepth = hpylmJSON.MaxDepth
	hpylm.theta = hpylmJSON.Theta
	hpylm.d = hpylmJSON.D
	hpylm.gammaA = hpylmJSON.GammaA
	hpylm.gammaB = hpylmJSON.GammaB
	hpylm.betaA = hpylmJSON.BetaA
	hpylm.betaB = hpylmJSON.BetaB
	hpylm.Base = hpylmJSON.Base

	return
}
