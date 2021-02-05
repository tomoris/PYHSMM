package bayselm

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"sync"
	"strings"

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
func NewPYHSMM(initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, PosSize int, splitter string) *PYHSMM {

	npylms := make([]*NPYLM, PosSize+1, PosSize+1)
	for pos := 0; pos < PosSize+1; pos++ {
		npylms[pos] = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, splitter)
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

	// pyhsmm.npylms[0].poissonCorrection()
	pyhsmm.npylms[0].vpylm.hpylm.estimateHyperPrameters()
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		pyhsmm.npylms[pos].estimateHyperPrameters()
	}
	pyhsmm.posHpylm.estimateHyperPrameters()
	return
}

// TestWordSegmentation inferences word segmentation and their POS tags from input unsegmented texts, and returns word sequence.
// This is used for common interface of NPYLM.
func (pyhsmm *PYHSMM) TestWordSegmentation(sents [][]string, threadsNum int) [][]string {
	wordSeqs, _ := pyhsmm.TestWordSegmentationAndPOSTagging(sents, threadsNum)
	return wordSeqs
}

// TestWordSegmentationForPython inferences word segmentation and their POS tags from input unsegmented texts, and returns data_container which contain segmented texts.
func (pyhsmm *PYHSMM) TestWordSegmentationForPython(sents [][]string, threadsNum int) *DataContainer {
	wordSeqs, _ := pyhsmm.TestWordSegmentationAndPOSTagging(sents, threadsNum)
	dataContainer := new(DataContainer)
	for _, wordSeq := range wordSeqs {
		dataContainer.SamplingWordSeqs = append(dataContainer.SamplingWordSeqs, wordSeq)
	}
	dataContainer.Size = len(wordSeqs)
	return dataContainer
}

// TestWordSegmentationAndPOSTagging inferences word segmentation and their POS tags from input unsegmented texts.
func (pyhsmm *PYHSMM) TestWordSegmentationAndPOSTagging(sents [][]string, threadsNum int) ([][]string, [][]int) {
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
				uPos[0] = strconv.Itoa(pyhsmm.bosPos)
				posP, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base)
				forwardScore[t][pos] = math.Log(p) + math.Log(posP)
			} else {
				forwardScoreTmp := make([]float64, 0, pyhsmm.PosSize)
				for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
					uPos[0] = strconv.Itoa(prevPos)
					posP, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base)
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

func (pyhsmm *PYHSMM) calcEachScoreForWord(sent []string) [][][][]float64 {
	// initialize eachScore
	type eachScoreForWordAndUAndPosType [][][][]float64
	eachScoreForWord := make(eachScoreForWordAndUAndPosType, len(sent), len(sent))
	for t := 0; t < len(sent); t++ {
		eachScoreForWord[t] = make([][][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			eachScoreForWord[t][k] = make([][]float64, pyhsmm.PosSize, pyhsmm.PosSize)
			for z := 0; z < pyhsmm.PosSize; z++ {
				eachScoreForWord[t][k][z] = make([]float64, pyhsmm.maxWordLength+1, pyhsmm.maxWordLength+1) // + 1 is for bos
				for j := 0; j < pyhsmm.maxWordLength+1; j++ {
					eachScoreForWord[t][k][z][j] = math.Inf(-1)
				}
			}
		}
	}

	word := string("")
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := float64(0.0)
	for t := 0; t < len(sent); t++ {
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			if t-k >= 0 {
				word = strings.Join(sent[(t-k):t+1], pyhsmm.npylms[0].splitter)
				base = pyhsmm.npylms[0].calcBase(word) // 文字レベルのスムージングは一つのVPYLMから
			} else {
				continue
			}
			for pos := 0; pos < pyhsmm.PosSize; pos++ {
				if t-k == 0 {
					u[0] = pyhsmm.bos
					uPos[0] = strconv.Itoa(pyhsmm.bosPos)
					wordScore, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
					score := math.Log(wordScore)
					if math.IsNaN(score) {
						errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), word (%v)", wordScore, word)
						panic(errMsg)
					}
					eachScoreForWord[t][k][pos][pyhsmm.maxWordLength] = score
					continue
				}
				for j := 0; j < pyhsmm.maxWordLength; j++ {
					if t-k-(j+1) >= 0 {
						u[0] = strings.Join(sent[(t-k-(j+1)):(t-k)], pyhsmm.npylms[0].splitter)
						wordScore, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
						score := math.Log(wordScore)
						if math.IsNaN(score) {
							errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), word (%v)", wordScore, word)
							panic(errMsg)
						}
						eachScoreForWord[t][k][pos][j] = score
					} else {
						continue
					}
				}
			}
		}
	}

	return eachScoreForWord
}

func (pyhsmm *PYHSMM) calcEachScoreForPos() [][]float64 {
	type eachScoreForPosType [][]float64
	eachScoreForPos := make(eachScoreForPosType, pyhsmm.PosSize, pyhsmm.PosSize+1)
	for pos := 0; pos < pyhsmm.PosSize; pos++ {
		eachScoreForPos[pos] = make([]float64, pyhsmm.PosSize+1, pyhsmm.PosSize+1)
	}

	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	for pos := 0; pos < pyhsmm.PosSize; pos++ {
		uPos[0] = strconv.Itoa(pyhsmm.bosPos)
		posScore, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base)
		score := math.Log(posScore)
		if math.IsNaN(score) {
			errMsg := fmt.Sprintf("forward error! score is NaN. posScore (%v), pos (%v), prevPos (%v)", posScore, pos, pyhsmm.bosPos)
			panic(errMsg)
		}
		eachScoreForPos[pos][pyhsmm.PosSize] = score

		for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
			uPos[0] = strconv.Itoa(prevPos)
			posScore, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base)
			score := math.Log(posScore)
			if math.IsNaN(score) {
				errMsg := fmt.Sprintf("forward error! score is NaN. posScore (%v), pos (%v), prevPos (%v)", posScore, pos, prevPos)
				panic(errMsg)
			}
			eachScoreForPos[pos][prevPos] = score
		}
	}

	return eachScoreForPos
}

func (pyhsmm *PYHSMM) forward(sent []string) forwardScoreForWordAndPosType {

	// initialize forwardScore
	forwardScore := make(forwardScoreForWordAndPosType, len(sent), len(sent))
	for t := 0; t < len(sent); t++ {
		forwardScore[t] = make([][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			forwardScore[t][k] = make([]float64, pyhsmm.PosSize, pyhsmm.PosSize)
		}
	}

	eachScoreForWord := pyhsmm.calcEachScoreForWord(sent)
	eachScoreForPos := pyhsmm.calcEachScoreForPos()

	for t := 0; t < len(sent); t++ {
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			if t-k >= 0 {
				//
			} else {
				continue
			}
			for pos := 0; pos < pyhsmm.PosSize; pos++ {
				if t-k == 0 {
					wordScoreLog := eachScoreForWord[t][k][pos][pyhsmm.maxWordLength]
					posScoreLog := eachScoreForPos[pos][pyhsmm.PosSize]
					score := wordScoreLog + posScoreLog
					if math.IsNaN(score) {
						errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v)", wordScoreLog, posScoreLog)
						panic(errMsg)
					}
					forwardScore[t][k][pos] = score
					continue
				}
				forwardScore[t][k][pos] = 0.0
				forwardScoreTmp := make([]float64, 0, pyhsmm.maxWordLength*pyhsmm.PosSize)
				for j := 0; j < pyhsmm.maxWordLength; j++ {
					if t-k-(j+1) >= 0 {
						//
					} else {
						continue
					}
					for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
						wordScoreLog := eachScoreForWord[t][k][pos][j]
						posScoreLog := eachScoreForPos[pos][prevPos]
						score := wordScoreLog + posScoreLog + forwardScore[t-(k+1)][j][prevPos]
						if math.IsNaN(score) {
							errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v)", wordScoreLog, posScoreLog)
							panic(errMsg)
						}
						forwardScoreTmp = append(forwardScoreTmp, score)
					}
				}

				logsumexpScore := pyhsmm.npylms[0].logsumexp(forwardScoreTmp)
				logsumexpScore = logsumexpScore - math.Log(float64(len(forwardScoreTmp)))
				if math.IsNaN(logsumexpScore) {
					errMsg := fmt.Sprintf("forward error! logsumexpScore is NaN. forwardScoreTmp (%v)", forwardScoreTmp)
					panic(errMsg)
				}
				forwardScore[t][k][pos] = logsumexpScore - math.Log(float64(len(forwardScoreTmp)))
			}
		}
	}

	// word := string("")
	// u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	// uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	// base := float64(0.0)
	// for t := 0; t < len(sent); t++ {
	// 	for k := 0; k < pyhsmm.maxWordLength; k++ {
	// 		if t-k >= 0 {
	// 			word = string(sent[(t - k) : t+1])
	// 			base = pyhsmm.npylms[0].calcBase(word) // 文字レベルのスムージングは一つのVPYLMから
	// 		} else {
	// 			continue
	// 		}
	// 		for pos := 0; pos < pyhsmm.PosSize; pos++ {
	// 			if t-k == 0 {
	// 				u[0] = pyhsmm.bos
	// 				uPos[0] = strconv.Itoa(pyhsmm.bosPos)
	// 				wordScore, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
	// 				posScore, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base)
	// 				score := math.Log(wordScore) + math.Log(posScore)
	// 				if math.IsNaN(score) {
	// 					errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v), word (%v)", wordScore, posScore, word)
	// 					panic(errMsg)
	// 				}
	// 				forwardScore[t][k][pos] = score
	// 				continue
	// 			}
	// 			forwardScore[t][k][pos] = 0.0
	// 			forwardScoreTmp := make([]float64, 0, pyhsmm.maxWordLength*pyhsmm.PosSize)
	// 			for j := 0; j < pyhsmm.maxWordLength; j++ {
	// 				if t-k-(j+1) >= 0 {
	// 					u[0] = string(sent[(t - k - (j + 1)):(t - k)])
	// 				} else {
	// 					continue
	// 				}
	// 				for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
	// 					wordScore, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
	// 					uPos[0] = strconv.Itoa(prevPos)
	// 					posScore, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base)
	// 					score := math.Log(wordScore) + math.Log(posScore) + forwardScore[t-(k+1)][j][prevPos]
	// 					if math.IsNaN(score) {
	// 						errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v), word (%v)", wordScore, posScore, word)
	// 						panic(errMsg)
	// 					}
	// 					forwardScoreTmp = append(forwardScoreTmp, score)
	// 				}
	// 			}

	// 			logsumexpScore := pyhsmm.npylms[0].logsumexp(forwardScoreTmp)
	// 			logsumexpScore = logsumexpScore - math.Log(float64(len(forwardScoreTmp)))
	// 			if math.IsNaN(logsumexpScore) {
	// 				errMsg := fmt.Sprintf("forward error! logsumexpScore is NaN. forwardScoreTmp (%v), word (%v)", forwardScoreTmp, word)
	// 				panic(errMsg)
	// 			}
	// 			forwardScore[t][k][pos] = logsumexpScore - math.Log(float64(len(forwardScoreTmp)))
	// 		}
	// 	}
	// }

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
			wordScore, _ := pyhsmm.npylms[prevPos].CalcProb(prevWord, u, base)
			uPos[0] = strconv.Itoa(nextPos)
			posScore, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(prevPos), uPos, pyhsmm.posHpylm.Base)
			score := math.Log(wordScore) + math.Log(posScore) + forwardScore[t-1][nextPos]
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

func (pyhsmm *PYHSMM) backward(sent []string, forwardScore forwardScoreForWordAndPosType, sampling bool) (context, []int) {
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
		maxScore := math.Inf(-1)
		maxJ := -1
		maxNextPos := -1
		for j := 0; j < pyhsmm.maxWordLength; j++ {
			for nextPos := 0; nextPos < pyhsmm.PosSize; nextPos++ {
				if t-k-(j+1) >= 0 {
					u[0] = strings.Join(sent[(t-k-(j+1)):(t-k)], pyhsmm.npylms[0].splitter)
					wordScore, _ := pyhsmm.npylms[prevPos].CalcProb(prevWord, u, base)
					uPos[0] = strconv.Itoa(nextPos)
					posScore, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(prevPos), uPos, pyhsmm.posHpylm.Base)
					score := math.Log(wordScore) + math.Log(posScore) + forwardScore[t-(k+1)][j][nextPos]
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
			sumScore := 0.0
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
		samplingWord = strings.Join(sent[(t-k-(j+1)):(t-k)], pyhsmm.npylms[0].splitter)
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
			uPos[0] = strconv.Itoa(pyhsmm.bosPos)
		} else {
			u[0] = wordSeq[i-1]
			uPos[0] = strconv.Itoa(posSeq[i-1])
		}
		// pyhsmm.npylms[pos].AddCustomer(word, u, base, pyhsmm.npylms[pos].addCustomerBase)
		pyhsmm.npylms[pos].AddCustomer(word, u, base, pyhsmm.npylms[0].addCustomerBase) // 文字レベルのスムージングは一つのVPYLMに追加
		pyhsmm.posHpylm.AddCustomer(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base, pyhsmm.posHpylm.addCustomerBaseNull)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	// base = pyhsmm.npylms[pyhsmm.eosPos].vpylm.hpylm.Base
	base = pyhsmm.npylms[0].vpylm.hpylm.Base
	pyhsmm.npylms[pyhsmm.eosPos].AddCustomer(pyhsmm.eos, u, base, pyhsmm.npylms[0].addCustomerBaseNull)
	uPos[0] = strconv.Itoa(posSeq[len(posSeq)-1])
	pyhsmm.posHpylm.AddCustomer(strconv.Itoa(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.Base, pyhsmm.posHpylm.addCustomerBaseNull)
	return
}

func (pyhsmm *PYHSMM) removeWordSeqAsCustomer(wordSeq context, posSeq []int) {
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	for i, word := range wordSeq {
		pos := posSeq[i]
		if i == 0 {
			u[0] = pyhsmm.bos
			uPos[0] = strconv.Itoa(pyhsmm.bosPos)
		} else {
			u[0] = wordSeq[i-1]
			uPos[0] = strconv.Itoa(posSeq[i-1])
		}
		pyhsmm.npylms[pos].RemoveCustomer(word, u, pyhsmm.npylms[0].removeCustomerBase)
		pyhsmm.posHpylm.RemoveCustomer(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.addCustomerBaseNull)
	}

	u[0] = wordSeq[len(wordSeq)-1]
	pyhsmm.npylms[pyhsmm.eosPos].RemoveCustomer(pyhsmm.eos, u, pyhsmm.npylms[0].removeCustomerBaseNull)
	uPos[0] = strconv.Itoa(posSeq[len(posSeq)-1])
	pyhsmm.posHpylm.RemoveCustomer(strconv.Itoa(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.addCustomerBaseNull)
	return
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
			samplingWordSeqs[i] = append(samplingWordSeqs[i], strings.Join(sent[start:end], pyhsmm.npylms[0].splitter))
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
func (pyhsmm *PYHSMM) InitializeFromAnnotatedData(dataContainer *DataContainer) {
	sents := dataContainer.Sents
	samplingWordSeqs := dataContainer.SamplingWordSeqs
	samplingPosSeqs := dataContainer.SamplingPosSeqs
	for i := 0; i < len(samplingWordSeqs); i++ {
		adjustedSamplingWordSeq := make(context, 0, len(sents[i]))
		adjustedSamplingPosSeq := make([]int, 0, len(sents[i]))
		for j := 0; j < len(samplingWordSeqs[i]); j++ {
			word := samplingWordSeqs[i][j]
			sliceWord := strings.Split(word, pyhsmm.npylms[0].splitter)
			wordLen := len(sliceWord)
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
					adjustedSamplingWordSeq = append(adjustedSamplingWordSeq, strings.Join(sliceWord[start:end], pyhsmm.npylms[0].splitter))
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
	removeFlag := false
	for pos := 0; pos < pyhsmm.PosSize; pos++ {
		if len(pyhsmm.npylms[pos].vpylm.hpylm.restaurants) != 0 { // epoch == 0
			removeFlag = true
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

	// pyhsmm.npylms[0].poissonCorrection()                  // 文字VPYLMは共通のものだけ
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
	pPosEos, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.Base)
	for pos := 0; pos < pyhsmm.PosSize; pos++ {
		// base := pyhsmm.npylms[pos].calcBase(word)
		pGivenPos, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
		pPos, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pos), uPos, pyhsmm.posHpylm.Base)
		p += pGivenPos * (pPos / (1.0 - pPosEos))
		sumPpos += pPos
	}
	return p + math.SmallestNonzeroFloat64
}

// ReturnMaxN returns maximum length of n-gram.
// This is used for interface of LmModel.
func (pyhsmm *PYHSMM) ReturnMaxN() int {
	return pyhsmm.maxNgram
}

// Save returns json.Marshal(pyhsmmJSON) and pyhsmmJSON.
// pyhsmmJSON is struct to save. its variables can be exported.
func (pyhsmm *PYHSMM) save() ([]byte, interface{}) {
	pyhsmmJSON := &pYHSMMJSON{
		Npylms: func(pyhsmm *PYHSMM) []*nPYLMJSON {
			npylmsJSON := make([]*nPYLMJSON, 0, len(pyhsmm.npylms))
			for _, npylm := range pyhsmm.npylms {
				_, npylmJSONInterface := npylm.save()
				npylmJSON, ok := npylmJSONInterface.(*nPYLMJSON)
				if !ok {
					panic("save error in PYHSMM")
				}
				npylmsJSON = append(npylmsJSON, npylmJSON)
			}
			return npylmsJSON
		}(pyhsmm),
		PosHpylm: func(pyhsmm *PYHSMM) *hPYLMJSON {
			_, posHpylmJSON := pyhsmm.posHpylm.save()
			return posHpylmJSON.(*hPYLMJSON)
		}(pyhsmm),

		MaxNgram:      pyhsmm.maxNgram,
		MaxWordLength: pyhsmm.maxWordLength,
		Bos:           pyhsmm.bos,
		Eos:           pyhsmm.eos,
		Bow:           pyhsmm.bow,
		Eow:           pyhsmm.eow,

		PosSize: pyhsmm.PosSize,
		EosPos:  pyhsmm.eosPos,
		BosPos:  pyhsmm.bosPos,
	}
	v, err := json.Marshal(&pyhsmmJSON)
	if err != nil {
		panic("save error in PYHSMM")
	}
	return v, pyhsmmJSON
}

// Load pyhsmm.
func (pyhsmm *PYHSMM) load(v []byte) {
	// 本当はposSizeだけのnpylmをスライスに格納させて読みこみたいが、面倒なので、適当に大きな値分のnpylmを作ってjsonを読む。
	// そうしないとエラー、最終的なモデルのnpylmsは適切な大きさで返される
	tmpPosSize := 100
	Npylms := make([]*nPYLMJSON, 0, tmpPosSize)
	for i := 0; i < tmpPosSize; i++ {
		npylmJSON := &nPYLMJSON{hPYLMJSON: &hPYLMJSON{Restaurants: make(map[string]*restaurantJSON)}}
		Npylms = append(Npylms, npylmJSON)
	}

	pyhsmmJSON := &pYHSMMJSON{Npylms: Npylms, PosHpylm: &hPYLMJSON{Restaurants: make(map[string]*restaurantJSON)}}
	err := json.Unmarshal(v, &pyhsmmJSON)
	if err != nil {
		panic("load error in PYHSMM")
	}
	pyhsmm.npylms = func(pyhsmmJSON *pYHSMMJSON) []*NPYLM {
		npylms := make([]*NPYLM, 0, 0)
		for _, npylmJSON := range pyhsmmJSON.Npylms {
			npylmV, err := json.Marshal(&npylmJSON)
			if err != nil {
				panic("load error in load npylm in PYHSMM")
			}
			// npylm := NewNPYLM(npylmJSON.Theta[0], npylmJSON.D[0], npylmJSON.GammaA[0], npylmJSON.GammaB[0], npylmJSON.BetaA[0], npylmJSON.BetaB[0], 0.1, 0.1, npylmJSON.MaxNgram, npylmJSON.MaxWordLength)
			// restaurants まで作ってあげないと、あとで nil pointer にアクセスしてエラーになる
			npylm := &NPYLM{HPYLM: &HPYLM{restaurants: make(map[string]*restaurant)}, vpylm: &VPYLM{hpylm: &HPYLM{restaurants: make(map[string]*restaurant)}}}
			npylm.load(npylmV)
			npylms = append(npylms, npylm)
		}
		return npylms
	}(pyhsmmJSON)

	posHpylmV, err := json.Marshal(&pyhsmmJSON.PosHpylm)
	if err != nil {
		panic("load error in load posHpylm in PYHSMM")
	}
	pyhsmm.posHpylm.load(posHpylmV)

	pyhsmm.maxNgram = pyhsmmJSON.MaxNgram
	pyhsmm.maxWordLength = pyhsmmJSON.MaxWordLength
	pyhsmm.bos = pyhsmmJSON.Bos
	pyhsmm.eos = pyhsmmJSON.Eos
	pyhsmm.bow = pyhsmmJSON.Bow
	pyhsmm.eow = pyhsmmJSON.Eow

	pyhsmm.PosSize = pyhsmmJSON.PosSize
	pyhsmm.eosPos = pyhsmmJSON.EosPos
	pyhsmm.bosPos = pyhsmmJSON.BosPos

	return
}

// EachScoreForPython is for python bindings.
type EachScoreForPython struct {
	eachScoreForWord [][][][][]float64
	eachScoreForPos  [][]float64
}

// GetWordScore returns log wordScore.
func (pyStrcut *EachScoreForPython) GetWordScore(i, t, k, z, j int) float64 {
	jj := j
	if j == -1 {
		jj = len(pyStrcut.eachScoreForWord[0][0])
	}

	if i >= len(pyStrcut.eachScoreForWord) {
		errMsg := fmt.Sprintf("GetScore error i %v, max is %v", i, len(pyStrcut.eachScoreForWord))
		panic(errMsg)
	}
	if t >= len(pyStrcut.eachScoreForWord[i]) {
		errMsg := fmt.Sprintf("GetScore error t %v, max is %v", t, len(pyStrcut.eachScoreForWord[i]))
		panic(errMsg)
	}
	if k >= len(pyStrcut.eachScoreForWord[i][t]) {
		errMsg := fmt.Sprintf("GetScore error k %v, max is %v", k, len(pyStrcut.eachScoreForWord[i][t]))
		panic(errMsg)
	}
	if z >= len(pyStrcut.eachScoreForWord[i][t][k]) {
		errMsg := fmt.Sprintf("GetScore error z %v, max is %v", z, len(pyStrcut.eachScoreForWord[i][t][k]))
		panic(errMsg)
	}
	if jj >= len(pyStrcut.eachScoreForWord[i][t][k][z]) {
		errMsg := fmt.Sprintf("GetScore error jj %v, max is %v", jj, len(pyStrcut.eachScoreForWord[i][t][k][z]))
		panic(errMsg)
	}

	wordScore := pyStrcut.eachScoreForWord[i][t][k][z][jj]
	if math.IsNaN(wordScore) {
		errMsg := fmt.Sprintf("calc error! score is NaN. wordScore (%v)", wordScore)
		panic(errMsg)
	}
	return wordScore
}

// GetPosScore returns log score.
func (pyStrcut *EachScoreForPython) GetPosScore(z, r int) float64 {
	rr := r
	if r == -1 {
		rr = len(pyStrcut.eachScoreForWord[0][0][0])
	}

	if z >= len(pyStrcut.eachScoreForPos) {
		errMsg := fmt.Sprintf("GetScore error z %v, max is %v", z, len(pyStrcut.eachScoreForPos))
		panic(errMsg)
	}
	if rr >= len(pyStrcut.eachScoreForPos[z]) {
		errMsg := fmt.Sprintf("GetScore error r %v, max is %v", r, len(pyStrcut.eachScoreForPos[z]))
		panic(errMsg)
	}

	posScore := pyStrcut.eachScoreForPos[z][rr]
	if math.IsNaN(posScore) {
		errMsg := fmt.Sprintf("calc error! score is NaN. posScore (%v)", posScore)
		panic(errMsg)
	}
	return posScore
}

// GetScore returns log score.
func (pyStrcut *EachScoreForPython) GetScore(i, t, k, z, j, r int) float64 {

	wordScore := pyStrcut.GetWordScore(i, t, k, z, j)
	posScore := pyStrcut.GetPosScore(z, r)
	score := wordScore + posScore
	if math.IsNaN(score) {
		errMsg := fmt.Sprintf("calc error! score is NaN. score (%v)", score)
		panic(errMsg)
	}
	return score
}

// GetEachScore returns forward score for python bindings.
func (pyhsmm *PYHSMM) GetEachScore(sents []string, threadsNum int) *EachScoreForPython {
	if pyhsmm.npylms[0].splitter != "" {
		panic("pyhsmm.npylms[0].splitter is not \"\"")
	}
	eachScore := &EachScoreForPython{eachScoreForWord: make([][][][][]float64, len(sents), len(sents)), eachScoreForPos: pyhsmm.calcEachScoreForPos()}
	ch := make(chan int, threadsNum)
	wg := sync.WaitGroup{}
	for i := range sents {
		ch <- 1
		wg.Add(1)
		go func(i int) {
			sentSlice := strings.Split(sents[i], pyhsmm.npylms[0].splitter)
			eachScoreForWord := pyhsmm.calcEachScoreForWord(sentSlice)
			eachScore.eachScoreForWord[i] = eachScoreForWord
			<-ch
			wg.Done()
		}(i)
	}
	wg.Wait()

	return eachScore
}

// TrainWithDiscScore trains parameter with forward score, which includes discriminator score, for python bindings.
func (pyhsmm *PYHSMM) TrainWithDiscScore(sent []string, logForwardScoreList []float64, dataContainer *DataContainer, index int, samping bool) {
	// load forwardScore
	i := 0
	forwardScore := make(forwardScoreForWordAndPosType, len(sent), len(sent))
	for t := 0; t < len(sent); t++ {
		forwardScore[t] = make([][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			forwardScore[t][k] = make([]float64, pyhsmm.PosSize, pyhsmm.PosSize)
			for z := 0; z < pyhsmm.PosSize; z++ {
				forwardScore[t][k][z] = logForwardScoreList[i]
				i++
			}
		}
	}

	sampledWordSeqs, sampledPosSeqs := pyhsmm.backward(sent, forwardScore, true)
	dataContainer.SamplingWordSeqs[index] = sampledWordSeqs
	dataContainer.SamplingPosSeqs[index] = sampledPosSeqs
	pyhsmm.addWordSeqAsCustomer(dataContainer.SamplingWordSeqs[index], dataContainer.SamplingPosSeqs[index])
	return
}

// CalcTestScore calculates score of word sequences score like perplixity.
func (pyhsmm *PYHSMM) CalcTestScore(wordSeqs [][]string, threadsNum int) (float64, float64) {
	// TODO
	return 0.0, 0.0
}

// ShowParameters shows hyperparameters of this model.
func (pyhsmm *PYHSMM) ShowParameters() {
	fmt.Println("estimated hyperparameters of PYHSMM")
	for pos := 0; pos < pyhsmm.PosSize+1; pos++ {
		fmt.Println("HPYLM", pos, "theta", pyhsmm.npylms[pos].theta)
		fmt.Println("HPYLM d", pos, pyhsmm.npylms[pos].d)
	}
	fmt.Println("VPYLM theta", pyhsmm.npylms[0].vpylm.hpylm.theta)
	fmt.Println("VPYLM d", pyhsmm.npylms[0].vpylm.hpylm.d)
	fmt.Println("VPYLM alpha", pyhsmm.npylms[0].vpylm.alpha)
	fmt.Println("VPYLM beta", pyhsmm.npylms[0].vpylm.beta)
	fmt.Println("posHpylm theta", pyhsmm.posHpylm.theta)
	fmt.Println("posHpylm d", pyhsmm.posHpylm.d)
}
