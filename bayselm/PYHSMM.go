package bayselm

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sync"

	"github.com/cheggaaa/pb/v3"
)

type forwardScoreForWordAndPosType [][][]float64

// PYHSMM contains posSize-th NPYLM instances.
type PYHSMM struct {
	npylms   []*NPYLM
	posHpylm *HPYLM

	maxNgram      int
	maxWordLength int
	bos           string
	eos           string
	bow           string
	eow           string

	PosSize int
	eosPos  int
	bosPos  int
}

// NewPYHSMM returns PYHSMM instance.
func NewPYHSMM(initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, PosSize int) *PYHSMM {

	npylms := make([]*NPYLM, PosSize+1, PosSize+1)
	for pos := 0; pos < PosSize+1; pos++ {
		npylms[pos] = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength)
	}
	posHpylm := NewHPYLM(maxNgram-1, initialTheta, initialD, gammaA, gammaB, betaA, betaB, 1.0/float64(PosSize+2))

	pyhsmm := &PYHSMM{npylms, posHpylm, maxNgram, maxWordLength, bos, "<EOS>", "<BOW>", "<EOW>", PosSize, PosSize, PosSize + 1}

	return pyhsmm
}

// TrainWordSegmentationAndPOSTagging trains word segentation model from unsegmnted texts without labeled data.
func (pyhsmm *PYHSMM) TrainWordSegmentationAndPOSTagging(dataContainer *DataContainer, threadsNum int, batchSize int) {
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
			pyhsmm.removeWordSeqAsCustomer(dataContainer.SamplingWordSeqs[r], dataContainer.SamplingPosSeqs[r])
		}
		sampledWordSeqs := make([]context, end-i, end-i)
		sampledPosSeqs := make([][]int, end-i, end-i)
		goldWordSeq := make(context, 0, 0) // dummy
		for j := i; j < end; j++ {
			ch <- 1
			wg.Add(1)
			go func(j int) {
				r := randIndexes[j]
				sent := dataContainer.Sents[r]
				forwardScore := pyhsmm.forward(sent)
				sampledWordSeqs[j-i], sampledPosSeqs[j-i] = pyhsmm.backward(sent, forwardScore, true, goldWordSeq)
				<-ch
				wg.Done()
			}(j)
		}
		wg.Wait()
		for j := i; j < end; j++ {
			r := randIndexes[j]
			dataContainer.SamplingWordSeqs[r] = sampledWordSeqs[j-i]
			dataContainer.SamplingPosSeqs[r] = sampledPosSeqs[j-i]
			pyhsmm.addWordSeqAsCustomer(dataContainer.SamplingWordSeqs[r], dataContainer.SamplingPosSeqs[r])
		}
	}
	bar.Finish()
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		pyhsmm.npylms[pos].estimateHyperPrameters()
		if pos != pyhsmm.eosPos {
			pyhsmm.npylms[pos].poissonCorrection()
			pyhsmm.npylms[pos].vpylm.hpylm.estimateHyperPrameters()
		}
	}
	pyhsmm.posHpylm.estimateHyperPrameters()
	return
}

// TestWordSegmentationAndPOSTagging inferences word segmentation from input unsegmented texts.
func (pyhsmm *PYHSMM) TestWordSegmentationAndPOSTagging(sents [][]rune, threadsNum int) ([][]string, [][]int) {
	wordSeqs := make([][]string, len(sents), len(sents))
	posSeqs := make([][]int, len(sents), len(sents))
	goldWordSeq := make(context, 0, 0) // dummy
	ch := make(chan int, threadsNum)
	wg := sync.WaitGroup{}
	for i := 0; i < len(sents); i++ {
		ch <- 1
		wg.Add(1)
		go func(i int) {
			forwardScore := pyhsmm.forward(sents[i])
			wordSeq, posSeq := pyhsmm.backward(sents[i], forwardScore, false, goldWordSeq)
			wordSeqs[i] = wordSeq
			posSeqs[i] = posSeq
			<-ch
			wg.Done()
		}(i)
	}
	wg.Wait()
	return wordSeqs, posSeqs
}

func (pyhsmm *PYHSMM) forward(sent []rune) forwardScoreForWordAndPosType {
	// initialize forwardScore
	forwardScore := make(forwardScoreForWordAndPosType, len(sent), len(sent))
	for t := 0; t < len(sent); t++ {
		forwardScore[t] = make([][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			forwardScore[t][k] = make([]float64, pyhsmm.PosSize, pyhsmm.PosSize)
		}
	}

	word := string("")
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := float64(0.0)
	for t := 0; t < len(sent); t++ {
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			for pos := 0; pos < pyhsmm.PosSize; pos++ {
				if t-k >= 0 {
					word = string(sent[(t - k) : t+1])
					u[0] = pyhsmm.bos
					uPos[0] = string(pyhsmm.bosPos)
					base = pyhsmm.npylms[pos].calcBase(word)
					if t-k == 0 {
						score, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
						posScore, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
						forwardScore[t][k][pos] = math.Log(score) + math.Log(posScore)
						continue
					}
				} else {
					continue
				}
				forwardScore[t][k][pos] = 0.0
				forwardScoreTmp := make([]float64, 0, pyhsmm.maxWordLength*pyhsmm.PosSize)
				for j := 0; j < pyhsmm.maxWordLength; j++ {
					for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
						if t-k-(j+1) >= 0 {
							u[0] = string(sent[(t - k - (j + 1)):(t - k)])
							score, _ := pyhsmm.npylms[prevPos].CalcProb(word, u, base)
							uPos[0] = string(prevPos)
							posScore, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
							score = math.Log(score) + math.Log(posScore) + forwardScore[t-(k+1)][j][prevPos]
							forwardScoreTmp = append(forwardScoreTmp, score)
						} else {
							continue
						}
					}

				}
				logsumexpScore := pyhsmm.npylms[0].logsumexp(forwardScoreTmp)
				forwardScore[t][k][pos] = logsumexpScore - float64(math.Log(float64(len(forwardScoreTmp))))
			}
		}
	}

	return forwardScore
}

func (pyhsmm *PYHSMM) backward(sent []rune, forwardScore forwardScoreForWordAndPosType, sampling bool, goldWordSeq context) (context, []int) {
	t := len(sent)
	k := 0
	prevWord := pyhsmm.eos
	prevPos := pyhsmm.eosPos
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := pyhsmm.npylms[0].vpylm.hpylm.Base
	samplingWord := string("")
	samplingWordSeq := make(context, 0, len(sent))
	samplingPosSeq := make([]int, 0, len(sent))
	samplingPosOnly := false
	wordCurrentPosition := -1
	if len(goldWordSeq) != 0 {
		samplingPosOnly = true
		wordCurrentPosition = len(goldWordSeq) - 1
	}
	for {
		if (t - k) == 0 {
			break
		}
		if prevWord != pyhsmm.eos {
			base = pyhsmm.npylms[prevPos].calcBase(prevWord)
		}
		// scoreArray := make([]float64, pyhsmm.maxWordLength*pyhsmm.PosSize, pyhsmm.maxWordLength*pyhsmm.PosSize)
		scoreArray := make([][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		// scoreArrayLog := make([][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		scoreArrayLog := make([]float64, pyhsmm.maxWordLength*pyhsmm.PosSize, pyhsmm.maxWordLength*pyhsmm.PosSize)
		for i := 0; i < pyhsmm.maxWordLength*pyhsmm.PosSize; i++ {
			scoreArrayLog[i] = math.Inf(-1)
		}
		maxScore := float64(math.Inf(-1))
		maxJ := -1
		maxNextPos := -1
		sumScore := float64(0.0)
		for j := 0; j < pyhsmm.maxWordLength; j++ {
			scoreArray[j] = make([]float64, pyhsmm.PosSize, pyhsmm.PosSize)
			if samplingPosOnly {
				if j != len([]rune(goldWordSeq[wordCurrentPosition]))-1 {
					continue
				}
			}
			for nextPos := 0; nextPos < pyhsmm.PosSize; nextPos++ {
				if t-k-(j+1) >= 0 {
					u[0] = string(sent[(t - k - (j + 1)):(t - k)])
					score, _ := pyhsmm.npylms[prevPos].CalcProb(prevWord, u, base)
					uPos[0] = string(nextPos)
					posScore, _ := pyhsmm.posHpylm.CalcProb(string(prevPos), uPos, pyhsmm.posHpylm.Base)
					score = math.Log(score) + math.Log(posScore) + forwardScore[t-(k+1)][j][nextPos]
					if score > maxScore {
						maxScore = score
						maxJ = j
						maxNextPos = nextPos
					}
					scoreArrayLog[j*pyhsmm.PosSize+nextPos] = score
					score = math.Exp(score)
					// scoreArray[j*pyhsmm.PosSize+nextPos] = score
					scoreArray[j][nextPos] = score
					sumScore += score
				} else {
					scoreArray[j][nextPos] = 0.0 // float64(math.Inf(-1))
					// scoreArrayLog[j*pyhsmm.PosSize+nextPos] = math.Inf(-1)
				}
			}
		}
		logSumScoreArrayLog := pyhsmm.npylms[0].logsumexp(scoreArrayLog)
		j := 0
		nextPos := 0
		if sampling {
			// r := rand.Float64() * sumScore
			r := rand.Float64()
			sumScore = 0.0
			for {
				sumScore += math.Exp(scoreArrayLog[j*pyhsmm.PosSize+nextPos] - logSumScoreArrayLog)
				// sumScore += scoreArray[j][nextPos]
				// sumScore += scoreArray[j*pyhsmm.PosSize+nextPos]
				if sumScore > r {
					break
				}
				nextPos++
				if nextPos == pyhsmm.PosSize {
					nextPos = 0
					j++
				}
				if j >= pyhsmm.maxWordLength {
					panic("sampling error in PYHSMM")
				}
			}
		} else {
			j = maxJ
			nextPos = maxNextPos
		}
		if t-k-(j+1) < 0 {
			panic("sampling error in PYHSMM")
		}
		if nextPos >= pyhsmm.PosSize {
			panic("sampling error in PYHSMM")
		}
		samplingWord = string(sent[(t - k - (j + 1)):(t - k)])
		samplingWordSeq = append(samplingWordSeq, samplingWord)
		samplingPosSeq = append(samplingPosSeq, nextPos)
		prevWord = samplingWord
		prevPos = nextPos
		t = t - (k + 1)
		k = j
		if samplingPosOnly {
			wordCurrentPosition--
		}
	}

	samplingWordReverse := make(context, len(samplingWordSeq), len(samplingWordSeq))
	samplingPosReverse := make([]int, len(samplingPosSeq), len(samplingPosSeq))
	for i, samplingWord := range samplingWordSeq {
		samplingWordReverse[(len(samplingWordSeq)-1)-i] = samplingWord
		samplingPosReverse[(len(samplingPosSeq)-1)-i] = samplingPosSeq[i]
	}
	if samplingPosOnly {
		if !reflect.DeepEqual(goldWordSeq, samplingWordReverse) {
			errMsg := fmt.Sprintf("backward error. samplingWordReverse (%v) does not match goldWordSeq (%v)", samplingWordReverse, goldWordSeq)
			panic(errMsg)
		}
	}
	return samplingWordReverse, samplingPosReverse
}

func (pyhsmm *PYHSMM) addWordSeqAsCustomer(wordSeq context, posSeq []int) {
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := float64(0.0)
	for i, word := range wordSeq {
		pos := posSeq[i]
		base = pyhsmm.npylms[pos].calcBase(word)
		if i == 0 {
			u[0] = pyhsmm.bos
			uPos[0] = string(pyhsmm.bosPos)
		} else {
			u[0] = wordSeq[i-1]
			u[0] = string(posSeq[i-1])
		}
		pyhsmm.npylms[pos].AddCustomer(word, u, base, pyhsmm.npylms[pos].addCustomerBase)
		pyhsmm.posHpylm.AddCustomer(string(pos), uPos, pyhsmm.posHpylm.Base, pyhsmm.posHpylm.addCustomerBaseNull)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	base = pyhsmm.npylms[pyhsmm.eosPos].vpylm.hpylm.Base
	pyhsmm.npylms[pyhsmm.eosPos].AddCustomer(pyhsmm.eos, u, base, pyhsmm.npylms[pyhsmm.eosPos].addCustomerBase)
	uPos[0] = string(posSeq[len(posSeq)-1])
	pyhsmm.posHpylm.AddCustomer(string(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.Base, pyhsmm.posHpylm.addCustomerBaseNull)
}

func (pyhsmm *PYHSMM) removeWordSeqAsCustomer(wordSeq context, posSeq []int) {
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	for i, word := range wordSeq {
		pos := posSeq[i]
		if i == 0 {
			u[0] = pyhsmm.bos
			uPos[0] = string(pyhsmm.bosPos)
		} else {
			u[0] = wordSeq[i-1]
			u[0] = string(posSeq[i-1])
		}
		pyhsmm.npylms[pos].RemoveCustomer(word, u, pyhsmm.npylms[pos].removeCustomerBase)
		pyhsmm.posHpylm.RemoveCustomer(string(pos), uPos, pyhsmm.posHpylm.addCustomerBaseNull)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	pyhsmm.npylms[pyhsmm.eosPos].RemoveCustomer(pyhsmm.eos, u, pyhsmm.npylms[pyhsmm.eosPos].removeCustomerBase)
	uPos[0] = string(posSeq[len(posSeq)-1])
	pyhsmm.posHpylm.RemoveCustomer(string(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.addCustomerBaseNull)
}

// Initialize initializes parameters.
func (pyhsmm *PYHSMM) Initialize(sents [][]rune, samplingWordSeqs []context, samplingPosSeqs [][]int) {
	for i := 0; i < len(sents); i++ {
		sent := sents[i]
		start := 0
		for {
			r := rand.Intn(pyhsmm.maxWordLength) + 1
			end := start + r
			if end > len(sent) {
				end = len(sent)
			}
			pos := rand.Intn(pyhsmm.PosSize)
			samplingWordSeqs[i] = append(samplingWordSeqs[i], string(sent[start:end]))
			samplingPosSeqs[i] = append(samplingPosSeqs[i], pos)
			start = end
			if start == len(sent) {
				break
			}
		}
		pyhsmm.addWordSeqAsCustomer(samplingWordSeqs[i], samplingPosSeqs[i])
	}
	return
}

// Train train n-gram parameters from given word sequences.
func (pyhsmm *PYHSMM) Train(dataContainer *DataContainer) {
	removeFlag := false
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		if len(pyhsmm.npylms[pos].vpylm.hpylm.restaurants) != 0 { // epoch == 0
			removeFlag = true
		}
	}
	randIndexes := rand.Perm(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i++ {
		r := randIndexes[i]
		wordSeq := dataContainer.SamplingWordSeqs[r]
		posSeq := dataContainer.SamplingPosSeqs[r]
		if removeFlag {
			pyhsmm.removeWordSeqAsCustomer(wordSeq, posSeq)
		}
		sent := dataContainer.Sents[r]
		forwardScore := pyhsmm.forward(sent)
		_, sampledPosSeq := pyhsmm.backward(sent, forwardScore, true, wordSeq)
		pyhsmm.addWordSeqAsCustomer(wordSeq, sampledPosSeq)
		dataContainer.SamplingPosSeqs[r] = sampledPosSeq
	}
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		pyhsmm.npylms[pos].estimateHyperPrameters()
		if pos != pyhsmm.eosPos {
			pyhsmm.npylms[pos].poissonCorrection()
			pyhsmm.npylms[pos].vpylm.hpylm.estimateHyperPrameters()
		}
	}
	pyhsmm.posHpylm.estimateHyperPrameters()
	return
}

// ReturnNgramProb returns n-gram probability.
// This is used for interface of LmModel.
func (pyhsmm *PYHSMM) ReturnNgramProb(word string, u context) float64 {
	p := 0.0
	for pos := 0; pos < pyhsmm.PosSize+2; pos++ {
		base := pyhsmm.npylms[pos].calcBase(word)
		pGivenPos, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
		uPos := context{""}
		pPos, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
		p += pGivenPos * pPos
	}
	return p
}

// ReturnMaxN returns maximum length of n-gram.
// This is used for interface of LmModel.
func (pyhsmm *PYHSMM) ReturnMaxN() int {
	return pyhsmm.maxNgram
}
