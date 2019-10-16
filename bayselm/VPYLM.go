package bayselm

import (
	"math"
	"math/rand"
	"strings"

	"github.com/cheggaaa/pb/v3"
)

// VPYLM contains n-gram parameters as restaurants in HPYLM instance.
type VPYLM struct {
	hpylm *HPYLM
	alpha float64 // hyper-parameter for beta distribution to estimate stop probability
	beta  float64 // hyper-parameter for beta distribution to estimate stop probability
}

// NewVPYLM returns VPYLM instance.
func NewVPYLM(maxDepth int, initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, base float64, alpha float64, beta float64) *VPYLM {
	vpylm := new(VPYLM)
	vpylm.hpylm = NewHPYLM(maxDepth, initialTheta, initialD, gammaA, gammaB, betaA, betaB, base)
	vpylm.alpha = float64(alpha)
	vpylm.beta = float64(beta)

	return vpylm
}

// AddCustomer adds n-gram parameters.
// n-gramの深さをサンプリングし、その深さに HPYLM.AddCustomer をしている。
func (vpylm *VPYLM) AddCustomer(word string, u context) int {
	depth := 0
	// if samplingDepth {
	_, _, probs := vpylm.CalcProb(word, u)
	sumScore := float64(0.0)
	for _, prob := range probs {
		sumScore += prob
	}

	// sampling depth
	r := float64(rand.Float64()) * sumScore
	sumScore = 0.0
	depth = 0
	for {
		sumScore += probs[depth]
		if sumScore > r {
			break
		}
		depth++
		if depth >= vpylm.hpylm.maxDepth+1 {
			panic("sampling error in VPYLM")
		}
	}
	vpylm.hpylm.AddCustomer(word, u[len(u)-depth:], vpylm.hpylm.Base, vpylm.hpylm.addCustomerBaseNull)
	vpylm.hpylm.addStopAndPassCount(word, u[len(u)-depth:])
	return depth
}

// RemoveCustomer removes n-gram parameters.
func (vpylm *VPYLM) RemoveCustomer(word string, u context, prevSampledDepth int) {
	// remove stops and passes
	vpylm.hpylm.removeStopAndPassCount(word, u[len(u)-prevSampledDepth:])
	vpylm.hpylm.RemoveCustomer(word, u[len(u)-prevSampledDepth:], vpylm.hpylm.removeCustomerBaseNull)
	return
}

// CalcProb returns n-gram prrobability.
// HPYLM と違い、context が与えられたときのすべての深さの確率を計算し、その値で各深さごとの n-gram 確率を重みづけする。
func (vpylm *VPYLM) CalcProb(word string, u context) (float64, []float64, []float64) {
	_, pNgrams := vpylm.hpylm.CalcProb(word, u, vpylm.hpylm.Base)

	stopProbs := make([]float64, len(u)+1, len(u)+1)
	vpylm.calcStopProbs(u, stopProbs)

	probs := make([]float64, len(u)+1, len(u)+1)
	pPass := float64(1.0)
	pStop := float64(1.0)
	p := float64(0.0)
	for i, pNgram := range pNgrams {
		pStop = stopProbs[i] * pPass
		p += pStop * pNgram
		probs[i] = pStop * pNgram

		pPass *= (1.0 - stopProbs[i])
	}

	return p + math.SmallestNonzeroFloat64, pNgrams, probs
}

func (vpylm *VPYLM) calcStopProbs(u context, stopProbs []float64) {
	if len(u) > vpylm.hpylm.maxDepth {
		panic("maximum depth error")
	}

	p := float64(0.0)
	stop := float64(0.0)
	pass := float64(0.0)
	for i := 0; i <= len(u); i++ {
		rst, ok := vpylm.hpylm.restaurants[strings.Join(u[i:], concat)]
		if ok {
			stop = float64(rst.stop)
			pass = float64(rst.pass)
		}
		p = (stop + vpylm.alpha) / (stop + pass + vpylm.alpha + vpylm.beta)
		stopProbs[len(u)-i] = p
	}
	return
}

// Train train n-gram parameters from given word sequences.
func (vpylm *VPYLM) Train(dataContainer *DataContainer) {
	removeFlag := true
	if len(vpylm.hpylm.restaurants) == 0 { // epoch == 0
		removeFlag = false
	}
	bar := pb.StartNew(dataContainer.Size)
	randIndexes := rand.Perm(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i++ {
		bar.Add(1)
		r := randIndexes[i]
		wordSeq := dataContainer.SamplingWordSeqs[r]
		if removeFlag {
			u := make(context, 0, vpylm.hpylm.maxDepth)
			for n := 0; n < vpylm.hpylm.maxDepth; n++ {
				u = append(u, bos)
			}
			for j, word := range wordSeq {
				vpylm.RemoveCustomer(word, u, dataContainer.SamplingDepthMemories[r][j])
				u = append(u[1:], word)
			}
		}
		sampledDepthMemory := make([]int, len(wordSeq), len(wordSeq))
		u := make(context, 0, vpylm.hpylm.maxDepth)
		for n := 0; n < vpylm.hpylm.maxDepth; n++ {
			u = append(u, bos)
		}
		for j, word := range wordSeq {
			sampledDepth := vpylm.AddCustomer(word, u)
			sampledDepthMemory[j] = sampledDepth
			u = append(u[1:], word)
		}
		dataContainer.SamplingDepthMemories[r] = sampledDepthMemory
	}
	bar.Finish()
	vpylm.hpylm.estimateHyperPrameters()
	return
}

// ReturnNgramProb returns n-gram probability.
// This is used for interface of LmModel.
func (vpylm *VPYLM) ReturnNgramProb(word string, u context) float64 {
	p, _, _ := vpylm.CalcProb(word, u)
	return p
}

// ReturnMaxN returns maximum length of n-gram.
// This is used for interface of LmModel.
func (vpylm *VPYLM) ReturnMaxN() int {
	return vpylm.hpylm.maxDepth + 1
}
