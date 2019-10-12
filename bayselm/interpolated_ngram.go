package bayselm

import (
	"fmt"
	"math"
	"strings"
)

// Ngram is n-gram struct.
type Ngram struct {
	contextToWordCounts map[string]map[string]int
	contextToCount      map[string]int

	maxN               int
	interporationRates []float64
	base               float64
	bos                string
}

// NewNgram returns new Ngram instance.
func NewNgram(maxN int, interporationRates []float64, base float64) *Ngram {
	ngram := new(Ngram)
	ngram.contextToWordCounts = make(map[string]map[string]int)
	ngram.contextToCount = make(map[string]int)
	ngram.interporationRates = make([]float64, 0, maxN)
	ngram.maxN = maxN
	if !(len(interporationRates) == ngram.maxN) {
		panic("length of interporationRates does not match maxN")
	}
	for i := 0; i < ngram.maxN; i++ {
		ngram.interporationRates[i] = interporationRates[i]
	}

	ngram.bos = "<BOS>"
	return ngram
}

// AddCount add word count and context count when n-gram is given.
func (ngram *Ngram) AddCount(word string, u context) {
	_, ok := ngram.contextToCount[strings.Join(u, concat)]
	if !ok {
		ngram.contextToCount[strings.Join(u, concat)] = 0
	}
	ngram.contextToCount[strings.Join(u, concat)]++
	wordCounts, ok := ngram.contextToWordCounts[strings.Join(u, concat)]
	if !ok {
		ngram.contextToWordCounts[strings.Join(u, concat)] = make(map[string]int)
		wordCounts, _ = ngram.contextToWordCounts[strings.Join(u, concat)]
	}

	_, ok = wordCounts[word]
	if !ok {
		wordCounts[word] = 0
	}
	wordCounts[word]++
	if len(u) != 0 {
		ngram.AddCount(word, u[1:])
	}
	return
}

// CalcProb returns n-gram prabability.
func (ngram *Ngram) CalcProb(word string, u context) float64 {
	if len(u) > ngram.maxN-1 {
		errMsg := fmt.Sprintf("CalcProb error. ngram (word = %v, context = %v) is longer than maxN (%v)", word, u, ngram.maxN)
		panic(errMsg)
	}
	contextCount := 0
	contextCount, ok := ngram.contextToCount[strings.Join(u, concat)]
	if !ok {
		contextCount = 0
	}
	wordCounts, ok := ngram.contextToWordCounts[strings.Join(u, concat)]
	wordCount, ok := wordCounts[word]
	if !ok {
		wordCount = 0
	}
	body := float64(0.0)
	if contextCount != 0 {
		body = float64(wordCount) / float64(contextCount)
	}

	lambda := ngram.interporationRates[len(u)]
	smoothing := 0.0
	if len(u) == 0 {
		smoothing = ngram.base
	} else {
		smoothing = ngram.CalcProb(word, u[1:])
	}
	p := (1.0-lambda)*body + lambda*smoothing
	return p
}

// TrainFromWordSeq train n-gram parameters from given word sequences.
func (ngram *Ngram) TrainFromWordSeq(dataContainer *DataContainer) {
	for i := 0; i < dataContainer.Size; i++ {
		wordSeq := dataContainer.SamplingWordSeqs[i]
		u := make(context, 0, ngram.maxN-1)
		for n := 0; n < ngram.maxN-1; n++ {
			u = append(u, ngram.bos)
		}
		for _, word := range wordSeq {
			ngram.AddCount(word, u)
			u = append(u[1:], word)
		}
	}
	return
}

// CalcPerplexity returns perplexity from input word sequence
func (ngram *Ngram) CalcPerplexity(wordSeq context) float64 {
	entropy := float64(0.0)
	u := make(context, 0, ngram.maxN-1)
	for n := 0; n < ngram.maxN-1; n++ {
		u = append(u, ngram.bos)
	}
	for _, word := range wordSeq {
		p := ngram.CalcProb(word, u)
		entropy += math.Log2(p)
		u = append(u[1:], word)
	}
	entropy *= -1
	entropy /= float64(len(wordSeq))
	perplexity := math.Exp2(entropy)
	return perplexity
}
