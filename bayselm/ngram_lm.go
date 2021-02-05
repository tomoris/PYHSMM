package bayselm

import (
	"encoding/json"
	"io/ioutil"
	"math"
)

const concat string = "<concat>"
const bos string = "<BOS>"

type newUint uint32

type context []string

// UnsupervisedWSM is unsupervised word segmentation model.
type UnsupervisedWSM interface {
	TrainWordSegmentation(*DataContainer, int, int)
	TestWordSegmentation([][]string, int) [][]string
	CalcTestScore([][]string, int) (float64, float64)
	Initialize(*DataContainer)
	InitializeFromAnnotatedData(*DataContainer)
	ShowParameters()
	save() ([]byte, interface{})
	load([]byte)
}

// GenerateUnsupervisedWSM returns UnsupervisedWSM instance.
func GenerateUnsupervisedWSM(modelName string, initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, PosSize int, base float64, splitter string) (UnsupervisedWSM, bool) {
	var model UnsupervisedWSM
	ok := false
	switch modelName {
	case "npylm":
		model = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, splitter)
		ok = true
	case "pyhsmm":
		model = NewPYHSMM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, PosSize, splitter)
		ok = true
	}
	return model, ok
}

// NgramLM is n-gram language model.
type NgramLM interface {
	Train(*DataContainer)
	ReturnNgramProb(string, context) float64
	ReturnMaxN() int
	save() ([]byte, interface{})
	load([]byte)
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
		model = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, "")
		ok = true
	case "pyhsmm":
		model = NewPYHSMM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, PosSize, "")
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

// Save model.
func Save(modelNgramLM NgramLM, saveFile string, saveFormat string) {
	// var modelNgramLM NgramLM
	// modelNgramLM = model.(NgramLM)
	modelJSONByte, modelJSON := modelNgramLM.save()
	if saveFormat == "indent" {
		var err error
		modelJSONByte, err = json.MarshalIndent(modelJSON, "", " ")
		if err != nil {
			panic("save model error")
		}
	} else if saveFormat == "notindent" {
		// pass
	} else {
		panic("save model error. please input corrent saveFormat")
	}
	err := ioutil.WriteFile(saveFile, modelJSONByte, 0644)
	if err != nil {
		panic("save model error")
	}
	return
}

// Load model.
func Load(modelName string, loadFile string) NgramLM {
	var model NgramLM
	ok := false
	// 以下のパラメータは後で更新されるので適当で良い
	initialTheta := 2.0
	initialD := 0.1
	gammaA := 0.1
	gammaB := 0.1
	betaA := 0.1
	betaB := 0.1
	alpha := 0.1
	beta := 0.1
	maxNgram := 2
	maxWordLength := 10
	PosSize := 10
	base := 0.1
	splitter := ""
	switch modelName {
	case "ngram":
		model = &Ngram{}
		ok = true
	case "hpylm":
		model = NewHPYLM(maxNgram-1, initialTheta, initialD, gammaA, gammaB, betaA, betaB, base)
		ok = true
	case "vpylm":
		model = NewVPYLM(maxNgram-1, initialTheta, initialD, gammaA, gammaB, betaA, betaB, base, alpha, beta)
		ok = true
	case "npylm":
		model = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, splitter)
		ok = true
	case "pyhsmm":
		model = NewPYHSMM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, PosSize, splitter)
		ok = true
	}
	if !ok {
		panic("load model initailize error")
	}
	modelJSONByte, err := ioutil.ReadFile(loadFile)
	if err != nil {
		panic("load model file error")
	}
	model.load(modelJSONByte)
	return model
}
