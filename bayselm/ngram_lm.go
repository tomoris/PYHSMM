package bayselm

import "math"

const concat string = "<concat>"
const bos string = "<BOS>"

type newUint uint32

type context []string

// LmModel is n-gram language model.
type LmModel interface {
	Train(*DataContainer)
	ReturnNgramProb(string, context) float64
	ReturnMaxN() int
}

// CalcPerplexity returns perplexity from input word sequence
func CalcPerplexity(model LmModel, dataContainer *DataContainer) float64 {
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
