package NPYLM

import (
	"math/rand"
	"strings"
)

type VPYLM struct {
	hpylm *HPYLM
	alpha newFloat // hyper-parameter for beta distribution to estimate stop probability
	beta  newFloat // hyper-parameter for beta distribution to estimate stop probability
}

func NewVPYLM(maxDepth int, initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, base float64, alpha float64, beta float64) *VPYLM {
	vpylm := new(VPYLM)
	vpylm.hpylm = NewHPYLM(maxDepth, initialTheta, initialD, gammaA, gammaB, betaA, betaB, base)
	vpylm.alpha = newFloat(alpha)
	vpylm.beta = newFloat(beta)

	return vpylm
}

func (vpylm *VPYLM) AddCustomer(word string, u context) int {
	depth := 0
	// if samplingDepth {
	_, _, probs := vpylm.CalcProb(word, u)
	sumScore := newFloat(0.0)
	for _, prob := range probs {
		sumScore += prob
	}

	// sampling depth
	r := newFloat(rand.Float64()) * sumScore
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
	// } else {
	// 	depth = len(u)
	// }

	// update stops and passes
	// dep := 0
	// for ; dep < depth; dep++ {
	// 	vpylm.passes[dep] += 1
	// }
	// vpylm.stops[dep] += 1
	// fmt.Println(depth, word, u, probs, "addCustmer from vpylm")
	vpylm.hpylm.AddCustomer(word, u[len(u)-depth:], vpylm.hpylm.Base, vpylm.hpylm.addCustomerBaseNull)
	vpylm.hpylm.AddStopAndPassCount(word, u[len(u)-depth:])
	return depth
}

func (vpylm *VPYLM) RemoveCustomer(word string, u context, prevSampledDepth int) {
	// remove stops and passes
	vpylm.hpylm.RemoveStopAndPassCount(word, u[len(u)-prevSampledDepth:])
	vpylm.hpylm.RemoveCustomer(word, u[len(u)-prevSampledDepth:], vpylm.hpylm.removeCustomerBaseNull)
	return
}

func (vpylm *VPYLM) CalcProb(word string, u context) (newFloat, []newFloat, []newFloat) {
	_, pNgrams := vpylm.hpylm.CalcProb(word, u, vpylm.hpylm.Base)

	stopProbs := make([]newFloat, len(u)+1, len(u)+1)
	vpylm.calcStopProbs(u, stopProbs)
	// fmt.Println(stopProbs, word, u, "CalcProb from vpylm")

	probs := make([]newFloat, len(u)+1, len(u)+1)
	pPass := newFloat(1.0)
	pStop := newFloat(1.0)
	p := newFloat(0.0)
	for i, pNgram := range pNgrams {
		pStop = stopProbs[i] * pPass
		p += pStop * pNgram
		probs[i] = pStop * pNgram

		pPass *= (1.0 - stopProbs[i])
	}

	return p, pNgrams, probs
}

func (vpylm *VPYLM) calcStopProbs(u context, stopProbs []newFloat) {
	if len(u) > vpylm.hpylm.maxDepth {
		panic("maximum depth error")
	}

	p := newFloat(0.0)
	stop := newFloat(0.0)
	pass := newFloat(0.0)
	for i := 0; i <= len(u); i++ {
		rst, ok := vpylm.hpylm.restaurants[strings.Join(u[i:], concat)]
		if ok {
			stop = newFloat(rst.stop)
			pass = newFloat(rst.pass)
		}
		p = (stop + vpylm.alpha) / (stop + pass + vpylm.alpha + vpylm.beta)
		stopProbs[len(u)-i] = p
	}
	return
}
