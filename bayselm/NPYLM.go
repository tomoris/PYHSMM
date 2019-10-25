package bayselm

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/cheggaaa/pb/v3"

	"gonum.org/v1/gonum/stat/distuv"
)

type forwardScoreType [][]float64

// NPYLM contains HPYLM instance as word-based n-gram parameters and VPYLM instance as character-based n-gram parameters.
type NPYLM struct {
	*HPYLM
	// add
	vpylm *VPYLM

	maxNgram      int
	maxWordLength int
	bos           string
	eos           string
	bow           string
	eow           string

	poisson     distuv.Poisson
	length2prob []float64

	word2sampledDepthMemory map[string][][]int
}

// NewNPYLM returns NPYLM instance.
func NewNPYLM(initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int) *NPYLM {

	charBase := float64(1.0 / 2097152.0) // 1 / 2^21 , size of character vocabulary in utf-8 encodeing
	dummyBase := charBase
	hpylm := NewHPYLM(maxNgram-1, initialTheta, initialD, gammaA, gammaB, betaA, betaB, dummyBase)
	vpylm := NewVPYLM(maxWordLength+2, initialTheta, initialD, gammaA, gammaB, betaA, betaB, charBase, alpha, beta)
	npylm := &NPYLM{hpylm, vpylm, maxNgram, maxWordLength, bos, "<EOS>", "<BOW>", "<EOW>", distuv.Poisson{}, make([]float64, maxWordLength, maxWordLength), make(map[string][][]int)}

	npylm.poisson.Lambda = float64(maxWordLength) / 2.0
	for k := 0; k < maxWordLength; k++ {
		npylm.length2prob[k] = 1.0 / float64(maxWordLength)
	}

	if npylm.maxNgram != 2 {
		panic("range of maxNgram is 2 to 2")
	}
	return npylm
}

func (npylm *NPYLM) addCustomerBase(word string) {
	if word != npylm.bos && word != npylm.eos {
		runeWord := []rune(word)
		sampledDepthMemory := make([]int, len(runeWord)+1, len(runeWord)+1)
		uChar := make(context, 0, npylm.maxWordLength) // +1 is for bos
		for i := 0; i < npylm.maxWordLength; i++ {
			uChar = append(uChar, npylm.bow)
		}
		for i := 0; i < len(runeWord); i++ {
			lastChar := string(runeWord[i])
			sampledDepth := npylm.vpylm.AddCustomer(lastChar, uChar)
			sampledDepthMemory[i] = sampledDepth
			uChar = append(uChar[1:], string(runeWord[i]))
		}
		sampledDepth := npylm.vpylm.AddCustomer(npylm.eow, uChar)
		sampledDepthMemory[len(runeWord)] = sampledDepth

		_, ok := npylm.word2sampledDepthMemory[word]
		if !ok {
			npylm.word2sampledDepthMemory[word] = [][]int{sampledDepthMemory}
		} else {
			npylm.word2sampledDepthMemory[word] = append(npylm.word2sampledDepthMemory[word], sampledDepthMemory)
		}

	}
	return
}

func (npylm *NPYLM) removeCustomerBase(word string) {
	if word != npylm.bos && word != npylm.eos {
		runeWord := []rune(word)
		sampledDepthMemories, ok := npylm.word2sampledDepthMemory[word]
		if !ok {
			errMsg := fmt.Sprintf("removeCustomerBase error. sampledDepthMemories of word (%v) does not exist", word)
			panic(errMsg)
		}
		if len(sampledDepthMemories) == 0 {
			errMsg := fmt.Sprintf("removeCustomerBase error. sampledDepthMemory of word (%v) does not exist", word)
			panic(errMsg)
		}
		sampledDepthMemory := sampledDepthMemories[0]
		uChar := make(context, 0, npylm.maxWordLength) // +1 is for bos
		for i := 0; i < npylm.maxWordLength; i++ {
			uChar = append(uChar, npylm.bow)
		}
		for i := 0; i < len(runeWord); i++ {
			lastChar := string(runeWord[i])
			npylm.vpylm.RemoveCustomer(lastChar, uChar, sampledDepthMemory[i])
			uChar = append(uChar[1:], string(runeWord[i]))
		}
		npylm.vpylm.RemoveCustomer(npylm.eow, uChar, sampledDepthMemory[len(runeWord)])

		npylm.word2sampledDepthMemory[word] = sampledDepthMemories[1:]
		if len(npylm.word2sampledDepthMemory[word]) == 0 {
			delete(npylm.word2sampledDepthMemory, word)
		}
	}
	return
}

func (npylm *NPYLM) calcBase(word string) float64 {
	p := float64(1.0)
	runeWord := []rune(word)
	// if len(runeWord) > npylm.maxWordLength {
	// 	errMsg := fmt.Sprintf("calcBase error. length of word (%v) is longer than npylm.maxWordLength (%v)", word, npylm.maxWordLength)
	// 	panic(errMsg)
	// }
	uChar := make(context, 0, npylm.maxWordLength) // +1 is for bos
	for i := 0; i < npylm.maxWordLength; i++ {
		uChar = append(uChar, npylm.bow)
	}
	for i := 0; i < len(runeWord); i++ {
		lastChar := string(runeWord[i])
		pTmpMixed, _, _ := npylm.vpylm.CalcProb(lastChar, uChar)
		p *= pTmpMixed
		uChar = append(uChar[1:], string(runeWord[i]))
	}
	pTmpMixed, _, _ := npylm.vpylm.CalcProb(npylm.eow, uChar)
	p *= pTmpMixed

	// poisson correction
	// if len(runeWord) <= npylm.maxWordLength {
	// 	p *= float64(npylm.poisson.Prob(float64(len(runeWord))) / float64(npylm.length2prob[len(runeWord)-1]))
	// }
	return p + math.SmallestNonzeroFloat64
}

func (npylm *NPYLM) logsumexp(forwardScoreTmp []float64) float64 {
	maxScore := math.Inf(-1)
	for _, score := range forwardScoreTmp {
		maxScore = math.Max(float64(score), maxScore)
	}

	logsumexpScoreTmp := 0.0
	for _, score := range forwardScoreTmp {
		logsumexpScoreTmp += math.Exp(float64(score) - maxScore)
	}
	logsumexpScore := float64(math.Log(logsumexpScoreTmp) + maxScore)

	return logsumexpScore
}

// TrainWordSegmentation trains word segentation model from unsegmnted texts without labeled data.
func (npylm *NPYLM) TrainWordSegmentation(dataContainer *DataContainer, threadsNum int, batchSize int) {
	ch := make(chan int, threadsNum)
	wg := sync.WaitGroup{}
	bar := pb.StartNew(dataContainer.Size)
	randIndexes := rand.Perm(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i += batchSize {
		end := i + batchSize
		if end > dataContainer.Size {
			end = dataContainer.Size
		}
		bar.Add(end - i)
		for j := i; j < end; j++ {
			r := randIndexes[j]
			npylm.removeWordSeqAsCustomer(dataContainer.SamplingWordSeqs[r])
		}
		sampledWordSeqs := make([]context, end-i, end-i)
		for j := i; j < end; j++ {
			ch <- 1
			wg.Add(1)
			go func(j int) {
				r := randIndexes[j]
				sent := dataContainer.Sents[r]
				forwardScore := npylm.forward(sent)
				sampledWordSeqs[j-i] = npylm.backward(sent, forwardScore, true)
				<-ch
				wg.Done()
			}(j)
		}
		wg.Wait()
		for j := i; j < end; j++ {
			r := randIndexes[j]
			dataContainer.SamplingWordSeqs[r] = sampledWordSeqs[j-i]
			npylm.addWordSeqAsCustomer(dataContainer.SamplingWordSeqs[r])
		}
	}
	bar.Finish()
	npylm.poissonCorrection()
	npylm.estimateHyperPrameters()
	npylm.vpylm.hpylm.estimateHyperPrameters()
	return
}

// TestWordSegmentation inferences word segmentation from input unsegmented texts.
func (npylm *NPYLM) TestWordSegmentation(sents [][]rune, threadsNum int) [][]string {
	wordSeqs := make([][]string, len(sents), len(sents))
	ch := make(chan int, threadsNum)
	if threadsNum <= 0 {
		panic("threadsNum should be bigger than 0")
	}
	wg := sync.WaitGroup{}
	for i := 0; i < len(sents); i++ {
		ch <- 1
		wg.Add(1)
		go func(i int) {
			forwardScore := npylm.forward(sents[i])
			wordSeq := npylm.backward(sents[i], forwardScore, false)
			wordSeqs[i] = wordSeq
			<-ch
			wg.Done()
		}(i)
	}
	wg.Wait()
	return wordSeqs
}

func (npylm *NPYLM) forward(sent []rune) forwardScoreType {
	// initialize forwardScore
	forwardScore := make(forwardScoreType, len(sent), len(sent))
	for t := 0; t < len(sent); t++ {
		forwardScore[t] = make([]float64, npylm.maxWordLength, npylm.maxWordLength)
	}

	word := string("")
	u := make(context, npylm.maxNgram-1, npylm.maxNgram-1) // now bi-gram only
	base := float64(0.0)
	for t := 0; t < len(sent); t++ {
		for k := 0; k < npylm.maxWordLength; k++ {
			if t-k >= 0 {
				word = string(sent[(t - k) : t+1])
				base = npylm.calcBase(word)
				if t-k == 0 {
					u[0] = npylm.bos
					score, _ := npylm.CalcProb(word, u, base)
					forwardScore[t][k] = math.Log(score)
					continue
				}
			} else {
				continue
			}
			forwardScore[t][k] = 0.0
			forwardScoreTmp := make([]float64, 0, npylm.maxWordLength)
			for j := 0; j < npylm.maxWordLength; j++ {
				if t-k-(j+1) >= 0 {
					u[0] = string(sent[(t - k - (j + 1)):(t - k)])
					score, _ := npylm.CalcProb(word, u, base)
					score = math.Log(score) + forwardScore[t-(k+1)][j]
					forwardScoreTmp = append(forwardScoreTmp, score)
				} else {
					continue
				}
			}
			logsumexpScore := npylm.logsumexp(forwardScoreTmp)
			forwardScore[t][k] = logsumexpScore - math.Log(float64(len(forwardScoreTmp)))
		}
	}

	return forwardScore
}

func (npylm *NPYLM) backward(sent []rune, forwardScore forwardScoreType, sampling bool) context {
	t := len(sent)
	k := 0
	prevWord := npylm.eos
	u := make(context, npylm.maxNgram-1, npylm.maxNgram-1)
	base := npylm.vpylm.hpylm.Base
	samplingWord := string("")
	samplingWordSeq := make(context, 0, len(sent))
	for {
		if (t - k) == 0 {
			break
		}
		if prevWord != npylm.eos {
			base = npylm.calcBase(prevWord)
		}
		scoreArrayLog := make([]float64, npylm.maxWordLength, npylm.maxWordLength)
		for j := 0; j < npylm.maxWordLength; j++ {
			scoreArrayLog[j] = math.Inf(-1)
		}
		maxScore := math.Inf(-1)
		maxJ := -1
		for j := 0; j < npylm.maxWordLength; j++ {
			if t-k-(j+1) >= 0 {
				u[0] = string(sent[(t - k - (j + 1)):(t - k)])
				score, _ := npylm.CalcProb(prevWord, u, base)
				score = math.Log(score) + forwardScore[t-(k+1)][j]
				if score > maxScore {
					maxScore = score
					maxJ = j
				}
				scoreArrayLog[j] = score
			} else {
				// scoreArray[j] = math.Inf(-1)
			}
		}
		logSumScoreArrayLog := npylm.logsumexp(scoreArrayLog)
		j := 0
		if sampling {
			r := rand.Float64()
			sumScore := 0.0
			for {
				score := math.Exp(scoreArrayLog[j] - logSumScoreArrayLog)
				sumScore += score
				if sumScore > r {
					break
				}
				j++
				if j >= npylm.maxWordLength {
					panic("sampling error in NPYLM")
				}
			}
		} else {
			j = maxJ
		}
		if t-k-(j+1) < 0 {
			panic("sampling error in NPYLM")
		}
		samplingWord = string(sent[(t - k - (j + 1)):(t - k)])
		samplingWordSeq = append(samplingWordSeq, samplingWord)
		prevWord = samplingWord
		t = t - (k + 1)
		k = j
	}

	samplingWordReverse := make(context, len(samplingWordSeq), len(samplingWordSeq))
	for i, samplingWord := range samplingWordSeq {
		samplingWordReverse[(len(samplingWordSeq)-1)-i] = samplingWord
	}
	return samplingWordReverse
}

func (npylm *NPYLM) addWordSeqAsCustomer(wordSeq context) {
	u := make(context, npylm.maxNgram-1, npylm.maxNgram-1)
	base := float64(0.0)
	for i, word := range wordSeq {
		base = npylm.calcBase(word)
		if i == 0 {
			u[0] = npylm.bos
		} else {
			u[0] = wordSeq[i-1]
		}
		npylm.AddCustomer(word, u, base, npylm.addCustomerBase)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	base = npylm.vpylm.hpylm.Base
	npylm.AddCustomer(npylm.eos, u, base, npylm.addCustomerBase)
}

func (npylm *NPYLM) removeWordSeqAsCustomer(wordSeq context) {
	u := make(context, npylm.maxNgram-1, npylm.maxNgram-1)
	for i, word := range wordSeq {
		if i == 0 {
			u[0] = npylm.bos
		} else {
			u[0] = wordSeq[i-1]
		}
		npylm.RemoveCustomer(word, u, npylm.removeCustomerBase)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	npylm.RemoveCustomer(npylm.eos, u, npylm.removeCustomerBase)
}

// Initialize initializes parameters.
func (npylm *NPYLM) Initialize(dataContainer *DataContainer) {
	sents := dataContainer.Sents
	samplingWordSeqs := dataContainer.SamplingWordSeqs
	for i := 0; i < len(sents); i++ {
		sent := sents[i]
		start := 0
		for {
			r := rand.Intn(npylm.maxWordLength) + 1
			end := start + r
			if end > len(sent) {
				end = len(sent)
			}
			samplingWordSeqs[i] = append(samplingWordSeqs[i], string(sent[start:end]))
			start = end
			if start == len(sent) {
				break
			}
		}
		npylm.addWordSeqAsCustomer(samplingWordSeqs[i])
	}
	return
}

// InitializeFromAnnotatedData initializes parameters from annotated texts.
func (npylm *NPYLM) InitializeFromAnnotatedData(sents [][]rune, samplingWordSeqs []context) {
	for i := 0; i < len(samplingWordSeqs); i++ {
		adjustedSamplingWordSeq := make(context, 0, len(sents[i]))
		for j := 0; j < len(samplingWordSeqs[i]); j++ {
			word := samplingWordSeqs[i][j]
			wordLen := len([]rune(word))
			if wordLen < npylm.maxWordLength {
				adjustedSamplingWordSeq = append(adjustedSamplingWordSeq, word)
			} else {
				start := 0
				for {
					end := start + npylm.maxWordLength
					if end > wordLen {
						end = wordLen
					}
					adjustedSamplingWordSeq = append(adjustedSamplingWordSeq, string([]rune(word)[start:end]))
					start = end
					if start == wordLen {
						break
					}
				}
			}
		}
		samplingWordSeqs[i] = adjustedSamplingWordSeq
		npylm.addWordSeqAsCustomer(samplingWordSeqs[i])
	}
	return
}

func (npylm *NPYLM) poissonCorrection() {
	_, ok := npylm.restaurants[""]
	if !ok {
		return
	}
	a := float64(1.0)
	b := float64(1.0)
	for word, totalTableCount := range npylm.restaurants[""].totalTableCountForCustomer {
		runeWord := []rune(word)
		a += (float64(totalTableCount) * float64(len(runeWord)))
		b += float64(totalTableCount)
	}
	g := distuv.Gamma{}
	g.Alpha = float64(a)
	g.Beta = float64(b)
	npylm.poisson.Lambda = g.Rand()

	length2count := make([]int, npylm.maxWordLength, npylm.maxWordLength)
	for k := 0; k < npylm.maxWordLength; k++ {
		length2count[k] = 1
	}

	charVocabSize := len(npylm.vpylm.hpylm.restaurants[""].totalTableCountForCustomer)
	chars := make([]string, 0, charVocabSize)
	for char := range npylm.vpylm.hpylm.restaurants[""].totalTableCountForCustomer {
		chars = append(chars, char)
	}
	sampleSize := 10000
	for i := 0; i < sampleSize; i++ {
		k := -1
		uChar := make(context, 0, npylm.maxWordLength) // +1 is for bos
		for i := 0; i < npylm.maxWordLength; i++ {
			uChar = append(uChar, npylm.bow)
		}
		for {
			probArray := make([]float64, charVocabSize, charVocabSize)
			sumScore := float64(0.0)
			for charIndex, char := range chars {
				if char == npylm.bow {
					continue
				}
				if char == npylm.eow && k == -1 {
					continue
				}
				prob, _, _ := npylm.vpylm.CalcProb(char, uChar)
				probArray[charIndex] = prob
				sumScore += prob
			}
			r := rand.Float64() * sumScore
			sumScore = 0.0
			charIndex := 0
			for _, prob := range probArray {
				sumScore += prob
				if sumScore > r {
					break
				}
				charIndex++
				if charIndex >= charVocabSize {
					panic("poissonCorrection error")
				}
			}
			char := chars[charIndex]

			if char == npylm.eow || k+1 >= npylm.maxWordLength {
				break
			}
			k++
			uChar = append(uChar[1:], char)
		}
		length2count[k]++
	}

	length2prob := make([]float64, npylm.maxWordLength, npylm.maxWordLength)
	for k := 0; k < npylm.maxWordLength; k++ {
		length2prob[k] = float64(length2count[k]) / (float64(sampleSize) + float64(npylm.maxWordLength))
	}

	npylm.length2prob = length2prob
	return
}

// Train train n-gram parameters from given word sequences.
func (npylm *NPYLM) Train(dataContainer *DataContainer) {
	removeFlag := true
	if len(npylm.vpylm.hpylm.restaurants) == 0 { // epoch == 0
		removeFlag = false
	}
	bar := pb.StartNew(dataContainer.Size)
	randIndexes := rand.Perm(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i++ {
		bar.Add(1)
		r := randIndexes[i]
		wordSeq := dataContainer.SamplingWordSeqs[r]
		if removeFlag {
			npylm.removeWordSeqAsCustomer(wordSeq)
		}
		npylm.addWordSeqAsCustomer(wordSeq)
	}
	bar.Finish()
	// npylm.poissonCorrection()
	npylm.estimateHyperPrameters()
	npylm.vpylm.hpylm.estimateHyperPrameters()
	return
}

// ReturnNgramProb returns n-gram probability.
// This is used for interface of LmModel.
func (npylm *NPYLM) ReturnNgramProb(word string, u context) float64 {
	base := npylm.calcBase(word)
	p, _ := npylm.CalcProb(word, u, base)
	return p
}

// ReturnMaxN returns maximum length of n-gram.
// This is used for interface of LmModel.
func (npylm *NPYLM) ReturnMaxN() int {
	return npylm.maxNgram
}

// Save returns json.Marshal(npylmJSON) and npylmJSON.
// npylmJSON is struct to save. its variables can be exported.
func (npylm *NPYLM) Save() ([]byte, interface{}) {
	npylmJSON := &nPYLMJSON{
		hPYLMJSON: &hPYLMJSON{Restaurants: func(rsts map[string]*restaurant) map[string]*restaurantJSON {
			rstsJSON := make(map[string]*restaurantJSON)
			for key, rst := range rsts {
				_, rstJSON := rst.save()
				rstsJSON[key] = rstJSON
			}
			return rstsJSON
		}(npylm.restaurants),

			MaxDepth: npylm.maxDepth,
			Theta:    npylm.theta,
			D:        npylm.d,
			GammaA:   npylm.gammaA,
			GammaB:   npylm.gammaB,
			BetaA:    npylm.betaA,
			BetaB:    npylm.betaB,
			Base:     npylm.Base,
		},

		Vpylm: func(npylm *NPYLM) *vPYLMJSON {
			_, vpylmJSONInterface := npylm.vpylm.Save()
			vpylmJSON, ok := vpylmJSONInterface.(*vPYLMJSON)
			if !ok {
				panic("save error in NPYLM")
			}
			return vpylmJSON
		}(npylm),

		MaxNgram:      npylm.maxNgram,
		MaxWordLength: npylm.maxWordLength,
		Bos:           npylm.bos,
		Eos:           npylm.eos,
		Bow:           npylm.bow,
		Eow:           npylm.eow,

		Poisson:     npylm.poisson,
		Length2prob: npylm.length2prob,
	}
	v, err := json.Marshal(&npylmJSON)
	if err != nil {
		panic("save error in NPYLM")
	}
	return v, npylmJSON
}

// Load npylm.
func (npylm *NPYLM) Load(v []byte) {
	npylmJSON := new(nPYLMJSON)
	err := json.Unmarshal(v, &npylmJSON)
	if err != nil {
		panic("load error in NPYLM")
	}

	// load npylm.restaurants
	// 一度map[string]*restaurantJSON を作ってから代入だとエラーになる (nil pointer)
	for key, rstJSON := range npylmJSON.Restaurants {
		rstV, err := json.Marshal(&rstJSON)
		if err != nil {
			panic("load error in load restaurants in HPYLM")
		}
		rst := newRestaurant()
		rst.load(rstV)
		npylm.restaurants[key] = rst
	}

	npylm.maxDepth = npylmJSON.MaxDepth
	npylm.theta = npylmJSON.Theta
	npylm.d = npylmJSON.D
	npylm.gammaA = npylmJSON.GammaA
	npylm.gammaB = npylmJSON.GammaB
	npylm.betaA = npylmJSON.BetaA
	npylm.betaB = npylmJSON.BetaB
	npylm.Base = npylmJSON.Base

	npylm.maxNgram = npylmJSON.MaxNgram
	npylm.maxWordLength = npylmJSON.MaxWordLength
	npylm.bos = npylmJSON.Bos
	npylm.eos = npylmJSON.Eos
	npylm.bow = npylmJSON.Bow
	npylm.eow = npylmJSON.Eow

	npylm.poisson = npylmJSON.Poisson
	npylm.length2prob = npylmJSON.Length2prob
	return
}
