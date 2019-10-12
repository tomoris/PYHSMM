package npylm

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"gonum.org/v1/gonum/stat/distuv"
)

const concat = "<concat>"

type newUint uint32

type context []string

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
	if initialD < 0.0 || initialD > 1.0 {
		panic("range of initialD is 0.0 to 1.0")
	}
	if initialTheta < 0.0 {
		panic("range of initialTheta is range 0.0 to inf")
	}
	if Base < 0.0 || Base > 1.0 {
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

// AddStopAndPassCount adds stop parameters.
// this function is used in VPYLM.
func (hpylm *HPYLM) AddStopAndPassCount(word string, u context) {
	rst, ok := hpylm.restaurants[strings.Join(u, concat)]
	if !ok {
		errMsg := fmt.Sprintf("AddStopAndPassCount error. context u (%v) does not exist", u)
		panic(errMsg)
	}
	rst.stop++
	for i := 1; i <= len(u); i++ {
		rst, ok := hpylm.restaurants[strings.Join(u[i:], concat)]
		if !ok {
			errMsg := fmt.Sprintf("AddStopAndPassCount error. context u (%v) does not exist", u[i:])
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
		score = float64(tbl) - (d * float64(rst.totalTableCountForCustomer[word]))
		scoreArray[k] = float64(math.Max(0.0, float64(score)))
		sumScore += scoreArray[k]
	}
	smoothingCoefficient := (theta + (d * float64(rst.totalTableCount))) / (theta + float64(rst.totalCustomerCount))
	if len(u) == 0 {
		scoreArray[tableNum] = smoothingCoefficient * base
	} else {
		scoreArray[tableNum] = smoothingCoefficient * probs[len(u)-1] // hpylm.CalcProb(word, u[1:])
	}
	sumScore += scoreArray[tableNum]

	// sampling
	r := float64(rand.Float64()) * sumScore
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
	r := float64(rand.Float64()) * sumScore
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

// RemoveStopAndPassCount removes stop parameters.
// this function is used in VPYLM.
func (hpylm *HPYLM) RemoveStopAndPassCount(word string, u context) {
	rst, ok := hpylm.restaurants[strings.Join(u, concat)]
	if !ok {
		errMsg := fmt.Sprintf("AddStopAndPassCount error. context u (%v) does not exist", u)
		panic(errMsg)
	}
	if rst.stop == 0 {
		errMsg := fmt.Sprintf("AddStopAndPassCount error. rst.stop of context u (%v) == 0", u)
		panic(errMsg)
	}
	rst.stop--

	for i := 1; i <= len(u); i++ {
		rst, ok := hpylm.restaurants[strings.Join(u[i:], concat)]
		if !ok {
			errMsg := fmt.Sprintf("AddStopAndPassCount error. context u (%v) does not exist", u[i:])
			panic(errMsg)
		}
		if rst.pass == 0 {
			errMsg := fmt.Sprintf("AddStopAndPassCount error. rst.stop of context u (%v) == 0", u)
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

	return p, probs
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
