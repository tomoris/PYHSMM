package bayselm

import (
	"fmt"
	"math"
	"math/rand"
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
	posHpylm := NewHPYLM(maxNgram-1, initialTheta, initialD, gammaA, gammaB, betaA, betaB, 1.0/float64(PosSize+1))

	pyhsmm := &PYHSMM{npylms, posHpylm, maxNgram, maxWordLength, bos, "<EOS>", "<BOW>", "<EOW>", PosSize, PosSize, PosSize + 1}

	return pyhsmm
}

// TrainWordSegmentation trains word segentation model and POS induction from unsegmnted texts without labeled data.
// This is used for common interface of NPYLM.
func (pyhsmm *PYHSMM) TrainWordSegmentation(dataContainer *DataContainer, threadsNum int, batchSize int) {
	pyhsmm.TrainWordSegmentationAndPOSTagging(dataContainer, threadsNum, batchSize)
	return
}

// TrainWordSegmentationAndPOSTagging trains word segentation model and POS induction from unsegmnted texts without labeled data.
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
		for j := i; j < end; j++ {
			ch <- 1
			wg.Add(1)
			go func(j int) {
				r := randIndexes[j]
				sent := dataContainer.Sents[r]
				forwardScore := pyhsmm.forward(sent)
				sampledWordSeqs[j-i], sampledPosSeqs[j-i] = pyhsmm.backward(sent, forwardScore, true)
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

	pyhsmm.npylms[0].poissonCorrection()
	pyhsmm.npylms[0].vpylm.hpylm.estimateHyperPrameters()
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		pyhsmm.npylms[pos].estimateHyperPrameters()
	}
	pyhsmm.posHpylm.estimateHyperPrameters()
	return
}

// TestWordSegmentation inferences word segmentation and their POS tags from input unsegmented texts, and returns word sequence.
// This is used for common interface of NPYLM.
func (pyhsmm *PYHSMM) TestWordSegmentation(sents [][]rune, threadsNum int) [][]string {
	wordSeqs, _ := pyhsmm.TestWordSegmentationAndPOSTagging(sents, threadsNum)
	return wordSeqs
}

// TestWordSegmentationAndPOSTagging inferences word segmentation and their POS tags from input unsegmented texts.
func (pyhsmm *PYHSMM) TestWordSegmentationAndPOSTagging(sents [][]rune, threadsNum int) ([][]string, [][]int) {
	wordSeqs := make([][]string, len(sents), len(sents))
	posSeqs := make([][]int, len(sents), len(sents))
	if threadsNum <= 0 {
		panic("threadsNum should be bigger than 0")
	}
	ch := make(chan int, threadsNum)
	wg := sync.WaitGroup{}
	for i := 0; i < len(sents); i++ {
		ch <- 1
		wg.Add(1)
		go func(i int) {
			forwardScore := pyhsmm.forward(sents[i])
			wordSeq, posSeq := pyhsmm.backward(sents[i], forwardScore, false)
			wordSeqs[i] = wordSeq
			posSeqs[i] = posSeq
			<-ch
			wg.Done()
		}(i)
	}
	wg.Wait()
	return wordSeqs, posSeqs
}

func (pyhsmm *PYHSMM) forwardForSamplingPosOnly(goldWordSeq context) [][]float64 {
	// initialize forwardScore
	// forwardScore[len(goldWordSeq)][posSize]
	forwardScore := make([][]float64, len(goldWordSeq), len(goldWordSeq))
	for t := 0; t < len(goldWordSeq); t++ {
		forwardScore[t] = make([]float64, pyhsmm.PosSize, pyhsmm.PosSize)
	}
	word := string("")
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := float64(0.0)
	for t := 0; t < len(goldWordSeq); t++ {
		for pos := 0; pos < pyhsmm.PosSize; pos++ {
			word = goldWordSeq[t]
			if t == 0 {
				u[0] = pyhsmm.bos
			} else {
			}
			// base = pyhsmm.npylms[pos].calcBase(word)
			base = pyhsmm.npylms[0].calcBase(word) // 文字レベルのスムージングは一つのVPYLMから
			p, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
			if t == 0 {
				uPos[0] = string(pyhsmm.bosPos)
				posP, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
				forwardScore[t][pos] = math.Log(p) + math.Log(posP)
			} else {
				forwardScoreTmp := make([]float64, 0, pyhsmm.PosSize)
				for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
					uPos[0] = string(prevPos)
					posP, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
					score := math.Log(p) + math.Log(posP) + forwardScore[t-1][prevPos]
					if math.IsNaN(score) {
						errMsg := fmt.Sprintf("forward error! score is NaN. p (%v), posP, (%v), word (%v)", p, posP, word)
						panic(errMsg)
					}
					forwardScoreTmp = append(forwardScoreTmp, score)
				}
				logsumexpScore := pyhsmm.npylms[0].logsumexp(forwardScoreTmp)
				if math.IsNaN(logsumexpScore) {
					errMsg := fmt.Sprintf("forward error! logsumexpScore is NaN. forwardScoreTmp (%v), word (%v)", forwardScoreTmp, word)
					panic(errMsg)
				}
				forwardScore[t][pos] = logsumexpScore
			}
		}
	}
	return forwardScore
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
			if t-k >= 0 {
				word = string(sent[(t - k) : t+1])
				base = pyhsmm.npylms[0].calcBase(word) // 文字レベルのスムージングは一つのVPYLMから
			}
			for pos := 0; pos < pyhsmm.PosSize; pos++ {
				if t-k >= 0 {
					u[0] = pyhsmm.bos
					uPos[0] = string(pyhsmm.bosPos)
					// base = pyhsmm.npylms[pos].calcBase(word)
					if t-k == 0 {
						wordScore, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
						posScore, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
						score := math.Log(wordScore) + math.Log(posScore)
						if math.IsNaN(score) {
							errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v), word (%v)", wordScore, posScore, word)
							panic(errMsg)
						}
						forwardScore[t][k][pos] = score
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
							wordScore, _ := pyhsmm.npylms[prevPos].CalcProb(word, u, base)
							uPos[0] = string(prevPos)
							posScore, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
							score := math.Log(wordScore) + math.Log(posScore) + forwardScore[t-(k+1)][j][prevPos]
							if math.IsNaN(score) {
								errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v), word (%v)", wordScore, posScore, word)
								panic(errMsg)
							}
							forwardScoreTmp = append(forwardScoreTmp, score)
						} else {
							continue
						}
					}

				}
				logsumexpScore := pyhsmm.npylms[0].logsumexp(forwardScoreTmp)
				if math.IsNaN(logsumexpScore - math.Log(float64(len(forwardScoreTmp)))) {
					errMsg := fmt.Sprintf("forward error! logsumexpScore is NaN. forwardScoreTmp (%v), word (%v)", forwardScoreTmp, word)
					panic(errMsg)
				}
				forwardScore[t][k][pos] = logsumexpScore - math.Log(float64(len(forwardScoreTmp)))
			}
		}
	}

	return forwardScore
}

func (pyhsmm *PYHSMM) backwardPosOnly(forwardScore [][]float64, sampling bool, goldWordSeq context) []int {
	t := len(goldWordSeq)
	prevWord := pyhsmm.eos
	prevPos := pyhsmm.eosPos
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := pyhsmm.npylms[0].vpylm.hpylm.Base
	samplingPosSeq := make([]int, 0, len(goldWordSeq))
	for {
		if t == 0 {
			break
		}
		if prevWord != pyhsmm.eos {
			base = pyhsmm.npylms[0].calcBase(prevWord) // 文字レベルのスムージングは一つのVPYLMから
		}
		scoreArrayLog := make([]float64, pyhsmm.PosSize, pyhsmm.PosSize)
		for i := 0; i < pyhsmm.PosSize; i++ {
			scoreArrayLog[i] = math.Inf(-1)
		}
		maxScore := float64(math.Inf(-1))
		maxNextPos := -1
		for nextPos := 0; nextPos < pyhsmm.PosSize; nextPos++ {
			u[0] = goldWordSeq[t-1]
			score, _ := pyhsmm.npylms[prevPos].CalcProb(prevWord, u, base)
			uPos[0] = string(nextPos)
			posScore, _ := pyhsmm.posHpylm.CalcProb(string(prevPos), uPos, pyhsmm.posHpylm.Base)
			score = math.Log(score) + math.Log(posScore) + forwardScore[t-1][nextPos]
			if score > maxScore {
				maxScore = score
				maxNextPos = nextPos
			}
			if math.IsNaN(score) {
				score = math.Inf(-1)
			}
			scoreArrayLog[nextPos] = score
		}
		logSumScoreArrayLog := pyhsmm.npylms[0].logsumexp(scoreArrayLog)
		nextPos := 0
		if sampling {
			r := rand.Float64()
			sumScore := 0.0
			for {
				sumScore += math.Exp(scoreArrayLog[nextPos] - logSumScoreArrayLog)
				if sumScore > r {
					break
				}
				nextPos++
				if nextPos >= pyhsmm.PosSize {
					panic("sampling error in PYHSMM")
				}
			}
		} else {
			nextPos = maxNextPos
		}
		samplingPosSeq = append(samplingPosSeq, nextPos)
		t--
		prevPos = nextPos
		prevWord = goldWordSeq[t]
	}
	samplingPosReverse := make([]int, len(samplingPosSeq), len(samplingPosSeq))
	for i := range samplingPosSeq {
		samplingPosReverse[(len(samplingPosSeq)-1)-i] = samplingPosSeq[i]
	}
	return samplingPosReverse
}

func (pyhsmm *PYHSMM) backward(sent []rune, forwardScore forwardScoreForWordAndPosType, sampling bool) (context, []int) {
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
	for {
		if (t - k) == 0 {
			break
		}
		if prevWord != pyhsmm.eos {
			base = pyhsmm.npylms[0].calcBase(prevWord) // 文字レベルのスムージングは一つのVPYLMから
		}
		scoreArrayLog := make([]float64, pyhsmm.maxWordLength*pyhsmm.PosSize, pyhsmm.maxWordLength*pyhsmm.PosSize)
		for i := 0; i < pyhsmm.maxWordLength*pyhsmm.PosSize; i++ {
			scoreArrayLog[i] = math.Inf(-1)
		}
		maxScore := float64(math.Inf(-1))
		maxJ := -1
		maxNextPos := -1
		sumScore := float64(0.0)
		for j := 0; j < pyhsmm.maxWordLength; j++ {
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
				} else {
					// scoreArrayLog[j*pyhsmm.PosSize+nextPos] = math.Inf(-1)
				}
			}
		}
		logSumScoreArrayLog := pyhsmm.npylms[0].logsumexp(scoreArrayLog)
		j := 0
		nextPos := 0
		if sampling {
			r := rand.Float64()
			sumScore = 0.0
			for {
				sumScore += math.Exp(scoreArrayLog[j*pyhsmm.PosSize+nextPos] - logSumScoreArrayLog)
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

func (pyhsmm *PYHSMM) addWordSeqAsCustomer(wordSeq context, posSeq []int) {
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := float64(0.0)
	for i, word := range wordSeq {
		pos := posSeq[i]
		// base = pyhsmm.npylms[pos].calcBase(word)
		base = pyhsmm.npylms[0].calcBase(word) // 文字レベルのスムージングは一つのVPYLMから
		if i == 0 {
			u[0] = pyhsmm.bos
			uPos[0] = string(pyhsmm.bosPos)
		} else {
			u[0] = wordSeq[i-1]
			u[0] = string(posSeq[i-1])
		}
		// pyhsmm.npylms[pos].AddCustomer(word, u, base, pyhsmm.npylms[pos].addCustomerBase)
		pyhsmm.npylms[pos].AddCustomer(word, u, base, pyhsmm.npylms[0].addCustomerBase) // 文字レベルのスムージングは一つのVPYLMに追加
		pyhsmm.posHpylm.AddCustomer(string(pos), uPos, pyhsmm.posHpylm.Base, pyhsmm.posHpylm.addCustomerBaseNull)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	// base = pyhsmm.npylms[pyhsmm.eosPos].vpylm.hpylm.Base
	base = pyhsmm.npylms[0].vpylm.hpylm.Base
	pyhsmm.npylms[pyhsmm.eosPos].AddCustomer(pyhsmm.eos, u, base, pyhsmm.npylms[0].addCustomerBase)
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
		pyhsmm.npylms[pos].RemoveCustomer(word, u, pyhsmm.npylms[0].removeCustomerBase)
		pyhsmm.posHpylm.RemoveCustomer(string(pos), uPos, pyhsmm.posHpylm.addCustomerBaseNull)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	pyhsmm.npylms[pyhsmm.eosPos].RemoveCustomer(pyhsmm.eos, u, pyhsmm.npylms[0].removeCustomerBase)
	uPos[0] = string(posSeq[len(posSeq)-1])
	pyhsmm.posHpylm.RemoveCustomer(string(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.addCustomerBaseNull)
}

// Initialize initializes parameters.
func (pyhsmm *PYHSMM) Initialize(dataContainer *DataContainer) {
	sents := dataContainer.Sents
	samplingWordSeqs := dataContainer.SamplingWordSeqs
	samplingPosSeqs := dataContainer.SamplingPosSeqs
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

// InitializeFromAnnotatedData initializes parameters from annotated texts.
func (pyhsmm *PYHSMM) InitializeFromAnnotatedData(sents [][]rune, samplingWordSeqs []context, samplingPosSeqs [][]int) {
	for i := 0; i < len(samplingWordSeqs); i++ {
		adjustedSamplingWordSeq := make(context, 0, len(sents[i]))
		adjustedSamplingPosSeq := make([]int, 0, len(sents[i]))
		for j := 0; j < len(samplingWordSeqs[i]); j++ {
			word := samplingWordSeqs[i][j]
			wordLen := len([]rune(word))
			if wordLen < pyhsmm.maxWordLength {
				adjustedSamplingWordSeq = append(adjustedSamplingWordSeq, word)
				adjustedSamplingPosSeq = append(adjustedSamplingPosSeq, samplingPosSeqs[i][j])
			} else {
				start := 0
				for {
					end := start + pyhsmm.maxWordLength
					if end > wordLen {
						end = wordLen
					}
					adjustedSamplingWordSeq = append(adjustedSamplingWordSeq, string([]rune(word)[start:end]))
					adjustedSamplingPosSeq = append(adjustedSamplingPosSeq, samplingPosSeqs[i][j])
					start = end
					if start == wordLen {
						break
					}
				}
			}
		}
		samplingWordSeqs[i] = adjustedSamplingWordSeq
		samplingPosSeqs[i] = adjustedSamplingPosSeq
		pyhsmm.addWordSeqAsCustomer(samplingWordSeqs[i], samplingPosSeqs[i])
	}
	return
}

// Train train n-gram parameters from given word sequences.
func (pyhsmm *PYHSMM) Train(dataContainer *DataContainer) {
	removeFlag := true
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		if len(pyhsmm.npylms[pos].vpylm.hpylm.restaurants) == 0 { // epoch == 0
			removeFlag = false
		}
	}
	bar := pb.StartNew(dataContainer.Size)
	randIndexes := rand.Perm(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i++ {
		bar.Add(1)
		r := randIndexes[i]
		wordSeq := dataContainer.SamplingWordSeqs[r]
		posSeq := dataContainer.SamplingPosSeqs[r]
		if removeFlag {
			pyhsmm.removeWordSeqAsCustomer(wordSeq, posSeq)
		}
		forwardScore := pyhsmm.forwardForSamplingPosOnly(wordSeq)
		sampledPosSeq := pyhsmm.backwardPosOnly(forwardScore, true, wordSeq)
		pyhsmm.addWordSeqAsCustomer(wordSeq, sampledPosSeq)
		dataContainer.SamplingPosSeqs[r] = sampledPosSeq
	}
	bar.Finish()

	pyhsmm.npylms[0].poissonCorrection()                  // 文字VPYLMは共通のものだけ
	pyhsmm.npylms[0].vpylm.hpylm.estimateHyperPrameters() // 文字VPYLMは共通のものだけ
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		pyhsmm.npylms[pos].estimateHyperPrameters()
	}
	pyhsmm.posHpylm.estimateHyperPrameters()
	return
}

// ReturnNgramProb returns n-gram probability.
// This is used for interface of LmModel.
// This func can not return correct probability if word is bos
func (pyhsmm *PYHSMM) ReturnNgramProb(word string, u context) float64 {
	p := 0.0
	sumPpos := 0.0
	base := pyhsmm.npylms[0].calcBase(word) // 文字レベルのスムージングは一つのVPYLMから
	uPos := context{""}
	pPosBos, _ := pyhsmm.posHpylm.CalcProb(string(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.Base)
	for pos := 0; pos < pyhsmm.PosSize; pos++ {
		// base := pyhsmm.npylms[pos].calcBase(word)
		pGivenPos, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
		pPos, _ := pyhsmm.posHpylm.CalcProb(string(pos), uPos, pyhsmm.posHpylm.Base)
		p += pGivenPos * (pPos / (1.0 - pPosBos))
		sumPpos += pPos
	}
	return p + math.SmallestNonzeroFloat64
}

// ReturnMaxN returns maximum length of n-gram.
// This is used for interface of LmModel.
func (pyhsmm *PYHSMM) ReturnMaxN() int {
	return pyhsmm.maxNgram
}
