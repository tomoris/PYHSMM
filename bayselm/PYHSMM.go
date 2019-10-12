// +build !

package bayselm

import (
	"fmt"
	"math"
	"math/rand"
	_ "strings"
)

type forwardScoreForWordAndPosType [][][]newFloat

type PYHSMM struct {
	npylms []*NPYLM

	maxNgram      int
	maxWordLength int
	bos           string
	eos           string
	bow           string
	eow           string

	eosPos int

	PosSize int
}

func NewPYHSMM(initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, PosSize int) *PYHSMM {

	npylms := make([]*NPYLM, PosSize+1, PosSize+1)
	for pos := 0; pos < PosSize+1; pos++ {
		npylms[pos] = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength)
	}

	pyhsmm := &PYHSMM{npylms, maxNgram, maxWordLength, "<BOS>", "<EOS>", "<BOW>", "<EOW>", PosSize, PosSize}

	return pyhsmm
}

func (pyhsmm *PYHSMM) Train(dataContainer *DataContainer) {
	for i := 0; i < dataContainer.Size; i++ {
		sent := dataContainer.Sents[i]
		pyhsmm.RemoveWordSeqAsCustomer(dataContainer.SamplingWordSeqs[i], dataContainer.SamplingPosSeqs[i])
		forwardScore := pyhsmm.forward(sent)
		dataContainer.SamplingWordSeqs[i], dataContainer.SamplingPosSeqs[i] = pyhsmm.backward(sent, forwardScore, true)
		fmt.Println(dataContainer.SamplingWordSeqs[i])
		fmt.Println(dataContainer.SamplingPosSeqs[i])
		pyhsmm.AddWordSeqAsCustomer(dataContainer.SamplingWordSeqs[i], dataContainer.SamplingPosSeqs[i])
	}
	for pos := 0; pos < pyhsmm.PosSize; pos++ {
		pyhsmm.npylms[pos].poissonCorrection()
		pyhsmm.npylms[pos].estimateHyperPrameters()
		pyhsmm.npylms[pos].vpylm.hpylm.estimateHyperPrameters()
	}
	return
}

func (pyhsmm *PYHSMM) Test(sent []rune) (context, []int) {
	forwardScore := pyhsmm.forward(sent)
	wordSeq, posSeq := pyhsmm.backward(sent, forwardScore, false)
	return wordSeq, posSeq
}

func (pyhsmm *PYHSMM) forward(sent []rune) forwardScoreForWordAndPosType {
	// initialize forwardScore
	forwardScore := make(forwardScoreForWordAndPosType, len(sent), len(sent))
	for t := 0; t < len(sent); t++ {
		forwardScore[t] = make([][]newFloat, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			forwardScore[t][k] = make([]newFloat, pyhsmm.PosSize, pyhsmm.PosSize)
		}
	}

	word := string("")
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := newFloat(0.0)
	for t := 0; t < len(sent); t++ {
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			for pos := 0; pos < pyhsmm.PosSize; pos++ {
				if t-k >= 0 {
					word = string(sent[(t - k) : t+1])
					u[0] = pyhsmm.bos
					base = pyhsmm.npylms[pos].calcBase(word)
					if t-k == 0 {
						score, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
						forwardScore[t][k][pos] = newFloat(math.Log(float64(score)))
						continue
					}
				} else {
					continue
				}
				forwardScore[t][k][pos] = 0.0
				forwardScoreTmp := make([]newFloat, 0, pyhsmm.maxWordLength*pyhsmm.PosSize)
				for j := 0; j < pyhsmm.maxWordLength; j++ {
					for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
						if t-k-(j+1) >= 0 {
							u[0] = string(sent[(t - k - (j + 1)):(t - k)])
							score, _ := pyhsmm.npylms[prevPos].CalcProb(word, u, base)
							score = newFloat(math.Log(float64(score)) + float64(forwardScore[t-(k+1)][j][prevPos]))
							forwardScoreTmp = append(forwardScoreTmp, score)
						} else {
							continue
						}
					}

				}
				logsumexpScore := pyhsmm.npylms[0].logsumexp(forwardScoreTmp)
				forwardScore[t][k][pos] = logsumexpScore
			}
		}
	}

	return forwardScore
}

func (pyhsmm *PYHSMM) backward(sent []rune, forwardScore forwardScoreForWordAndPosType, sampling bool) (context, []int) {
	t := len(sent)
	k := 0
	prevWord := pyhsmm.eos
	prevPos := pyhsmm.eosPos
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := pyhsmm.npylms[0].vpylm.hpylm.Base
	samplingWord := string("")
	samplingWordSeq := make(context, 0, len(sent))
	samplingPosSeq := make([]int, 0, len(sent))
	for {
		if (t - k) == 0 {
			break
		}
		if prevWord != pyhsmm.eos {
			base = pyhsmm.npylms[prevPos].calcBase(prevWord)
		}
		scoreArray := make([]newFloat, pyhsmm.maxWordLength*pyhsmm.PosSize, pyhsmm.maxWordLength*pyhsmm.PosSize)
		maxScore := newFloat(math.Inf(-1))
		maxJ := -1
		maxNextPos := -1
		sumScore := newFloat(0.0)
		for j := 0; j < pyhsmm.maxWordLength; j++ {
			for nextPos := 0; nextPos < pyhsmm.PosSize; nextPos++ {
				if t-k-(j+1) >= 0 {
					u[0] = string(sent[(t - k - (j + 1)):(t - k)])
					score, _ := pyhsmm.npylms[prevPos].CalcProb(prevWord, u, base)
					score = newFloat(math.Log(float64(score)) + float64(forwardScore[t-(k+1)][j][nextPos]))
					if score > maxScore {
						maxScore = score
						maxJ = j
						maxNextPos = nextPos
					}
					score = newFloat(math.Exp(float64(score)))
					scoreArray[j*pyhsmm.PosSize+nextPos] = score
					sumScore += score
				} else {
					scoreArray[j*pyhsmm.PosSize+nextPos] = newFloat(math.Inf(-1))
				}
			}
		}
		j := 0
		nextPos := 0
		if sampling {
			r := newFloat(rand.Float64()) * sumScore
			sumScore = 0.0
			for {
				sumScore += scoreArray[j*pyhsmm.PosSize+nextPos]
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
	}

	samplingWordReverse := make(context, len(samplingWordSeq), len(samplingWordSeq))
	samplingPosReverse := make([]int, len(samplingPosSeq), len(samplingPosSeq))
	for i, samplingWord := range samplingWordSeq {
		samplingWordReverse[(len(samplingWordSeq)-1)-i] = samplingWord
		samplingPosReverse[(len(samplingPosSeq)-1)-i] = samplingPosSeq[i]
	}
	return samplingWordReverse, samplingPosReverse
}

func (pyhsmm *PYHSMM) AddWordSeqAsCustomer(wordSeq context, posSeq []int) {
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := newFloat(0.0)
	for i, word := range wordSeq {
		pos := posSeq[i]
		base = pyhsmm.npylms[pos].calcBase(word)
		if i == 0 {
			u[0] = pyhsmm.bos
		} else {
			u[0] = wordSeq[i-1]
		}
		pyhsmm.npylms[pos].AddCustomer(word, u, base, pyhsmm.npylms[pos].addCustomerBase)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	base = pyhsmm.npylms[pyhsmm.eosPos].vpylm.hpylm.Base
	pyhsmm.npylms[pyhsmm.eosPos].AddCustomer(pyhsmm.eos, u, base, pyhsmm.npylms[pyhsmm.eosPos].addCustomerBase)
}

func (pyhsmm *PYHSMM) RemoveWordSeqAsCustomer(wordSeq context, posSeq []int) {
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	for i, word := range wordSeq {
		pos := posSeq[i]
		if i == 0 {
			u[0] = pyhsmm.bos
		} else {
			u[0] = wordSeq[i-1]
		}
		pyhsmm.npylms[pos].RemoveCustomer(word, u, pyhsmm.npylms[pos].removeCustomerBase)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	pyhsmm.npylms[pyhsmm.eosPos].RemoveCustomer(pyhsmm.eos, u, pyhsmm.npylms[pyhsmm.eosPos].removeCustomerBase)
}

func (pyhsmm *PYHSMM) Initialize(sents [][]rune, samplingWordSeqs []context, samplingPosSeqs [][]int) {
	for i := 0; i < len(sents); i++ {
		sent := sents[i]
		start := 0
		for {
			end := start + pyhsmm.maxWordLength
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
		pyhsmm.AddWordSeqAsCustomer(samplingWordSeqs[i], samplingPosSeqs[i])
	}
	return
}
