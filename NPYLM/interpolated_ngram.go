
type context []string

type Ngram struct {
	contextToWordCount map[string]map[string]int

	maxDepth int
	interporationRates []float64
	base float64
}

func NewHPYLM(maxDepth int interporationRates []float64, base) *HPYLM {
	ngram := new(Ngram)
	ngram.contextToWordCount = make(map[string]map[string]int)
	ngram.interporationRates = make([]float64, 0, maxDepth)
	ngram.maxDepth = maxDepth
	if !(len(interporationRates) == ngram.maxDepth) {
		panic("length of interporationRates does not match maxDepth")
	}
	for i := 0; i <= ngram.maxDepth; i++ {
		ngram.interporationRates[i] = interporationRates[i]
	}

	return ngram
}

func (hpylm *HPYLM) AddCount(word string, u context) {

	return
}