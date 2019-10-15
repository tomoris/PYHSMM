package bayselm

import "math"

const concat string = "<concat>"
const bos string = "<BOS>"

type newUint uint32

type context []string

// NgramLM is n-gram language model.
type NgramLM interface {
	Train(*DataContainer)
	ReturnNgramProb(string, context) float64
	ReturnMaxN() int
}

// GenerateNgramLM returns NgramLM instance.
func GenerateNgramLM(modelName string, initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, PosSize int, base float64) (NgramLM, bool) {
	var model NgramLM
	ok := false
	switch modelName {
	case "ngram":
		var interporationRates []float64
		for i := 0; i < maxNgram; i++ {
			interporationRates = append(interporationRates, 0.1)
		}
		model = NewNgram(maxNgram, interporationRates, base)
		ok = true
	case "hpylm":
		model = NewHPYLM(maxNgram-1, initialTheta, initialD, gammaA, gammaB, betaA, betaB, base)
		ok = true
	case "vpylm":
		model = NewVPYLM(maxNgram-1, initialTheta, initialD, gammaA, gammaB, betaA, betaB, base, alpha, beta)
		ok = true
	case "npylm":
		model = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength)
		ok = true
	case "pyhsmm":
		model = NewPYHSMM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, PosSize)
		ok = true
	}
	return model, ok
}

// CalcPerplexity returns perplexity from input word sequence
func CalcPerplexity(model NgramLM, dataContainer *DataContainer) float64 {
	entropy := float64(0.0)
	countWord := 0
	maxN := model.ReturnMaxN()
	for i := 0; i < dataContainer.Size; i++ {
		wordSeq := dataContainer.SamplingWordSeqs[i]
		u := make(context, 0, maxN-1)
		for n := 0; n < maxN-1; n++ {
			u = append(u, bos)
		}
		for _, word := range wordSeq {
			p := model.ReturnNgramProb(word, u)
			entropy += math.Log2(p)
			u = append(u[1:], word)
		}
		countWord += len(wordSeq)
	}
	entropy *= -1
	entropy /= float64(countWord)
	perplexity := math.Exp2(entropy)

	return perplexity
}
