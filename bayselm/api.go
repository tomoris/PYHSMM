package bayselm

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"sync"

	"github.com/cheggaaa/pb/v3"
)

// APIParam .
type APIParam struct {
	SentIDs       []int
	Sents         []string
	ForwardScores []forwardScoreForWordAndPosType
	LowerBound    float64
	ThreadsNum    int
	Lambda0       float64
	DiscScores    [][][][]float64
	DiscScoreT    [][]float64
}

// GetPYHSMMFeatsAPI .
func GetPYHSMMFeatsAPI(pyhsmm *PYHSMM, dataContainer *DataContainer, apiParam APIParam) []GenerativeFeatures {
	gFeatsSlice := make([]GenerativeFeatures, len(apiParam.SentIDs), len(apiParam.SentIDs))
	ch := make(chan int, apiParam.ThreadsNum)
	wg := sync.WaitGroup{}
	if apiParam.ThreadsNum == 0 {
		panic("ThreadsNum == 0")
	}
	for i, sentID := range apiParam.SentIDs {
		ch <- 1
		wg.Add(1)
		go func(i int, sentID int) {
			sent := dataContainer.Sents[sentID]
			gFeatsSlice[i] = pyhsmm.getGenerativeFeatures(sent, apiParam.LowerBound)
			<-ch
			wg.Done()
		}(i, sentID)
	}
	wg.Wait()
	if len(gFeatsSlice) != len(apiParam.SentIDs) {
		panic("len(gFeatsSlice) != len(apiParam.SentIDs")
	}
	adjustGFeatsSlice := adjustGFeatsSlice(pyhsmm, gFeatsSlice, apiParam)
	return adjustGFeatsSlice
}

// GetPYHSMMFeatsFromSentsAPI .
func GetPYHSMMFeatsFromSentsAPI(pyhsmm *PYHSMM, dataContainer *DataContainer, apiParam APIParam) []GenerativeFeatures {
	gFeatsSlice := make([]GenerativeFeatures, len(apiParam.Sents), len(apiParam.Sents))
	if apiParam.ThreadsNum == 0 {
		panic("ThreadsNum == 0")
	}
	ch := make(chan int, apiParam.ThreadsNum)
	wg := sync.WaitGroup{}
	for i, sent := range apiParam.Sents {
		ch <- 1
		wg.Add(1)
		go func(i int, sent string) {
			newSent := strings.Split(sent, pyhsmm.npylms[0].splitter)
			gFeatsSlice[i] = pyhsmm.getGenerativeFeatures(newSent, apiParam.LowerBound)
			<-ch
			wg.Done()
		}(i, sent)

	}
	wg.Wait()
	if len(gFeatsSlice) != len(apiParam.Sents) {
		panic("len(gFeatsSlice) != len(apiParam.Sents")
	}
	adjustGFeatsSlice := adjustGFeatsSlice(pyhsmm, gFeatsSlice, apiParam)
	return adjustGFeatsSlice
}

func validAPIParam(pyhsmm *PYHSMM, dataContainer *DataContainer, apiParam APIParam) {
	for i, sentID := range apiParam.SentIDs {
		sent := dataContainer.Sents[sentID]
		if len(sent) != len(apiParam.ForwardScores[i]) {
			fmt.Println(len(sent), len(apiParam.ForwardScores[i]))
			panic("len(sent) != len(apiParam.ForwardScores[i])")
		}
	}
}

// AddCustomerUsingForwardScoreAPI .
func AddCustomerUsingForwardScoreAPI(pyhsmm *PYHSMM, dataContainer *DataContainer, apiParam APIParam) {
	validAPIParam(pyhsmm, dataContainer, apiParam)
	for i, sentID := range apiParam.SentIDs {
		sent := dataContainer.Sents[sentID]
		forwardScore := apiParam.ForwardScores[i]
		sampledWordSeqs, sampledPosSeqs := pyhsmm.backwardJESSCM(sent, forwardScore, true, apiParam.Lambda0, apiParam.DiscScores[i], apiParam.DiscScoreT, apiParam.LowerBound)
		dataContainer.SamplingWordSeqs[sentID] = sampledWordSeqs
		dataContainer.SamplingPosSeqs[sentID] = sampledPosSeqs
		pyhsmm.addWordSeqAsCustomer(dataContainer.SamplingWordSeqs[sentID], dataContainer.SamplingPosSeqs[sentID])
	}
}

// RemoveCustomerAPI .
func RemoveCustomerAPI(pyhsmm *PYHSMM, dataContainer *DataContainer, apiParam APIParam) {
	for _, sentID := range apiParam.SentIDs {
		wordSeq := dataContainer.SamplingWordSeqs[sentID]
		posSeq := dataContainer.SamplingPosSeqs[sentID]
		pyhsmm.removeWordSeqAsCustomer(wordSeq, posSeq)
	}
}

// TrainFromAnnotatedCorpus .
func TrainFromAnnotatedCorpus(pyhsmm *PYHSMM, dataContainer *DataContainer) {
	// remove and add
	bar := pb.StartNew(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i++ {
		bar.Add(1)
		wordSeq := dataContainer.SamplingWordSeqs[i]
		posSeq := dataContainer.SamplingPosSeqs[i]
		pyhsmm.removeWordSeqAsCustomer(wordSeq, posSeq)
		pyhsmm.addWordSeqAsCustomer(wordSeq, posSeq)
	}
	bar.Finish()
}

// AddWordSeqAsCustomerAPI .
func AddWordSeqAsCustomerAPI(pyhsmm *PYHSMM, dataContainer *DataContainer) {
	bar := pb.StartNew(dataContainer.Size)
	for i := 0; i < dataContainer.Size; i++ {
		bar.Add(1)
		wordSeq := dataContainer.SamplingWordSeqs[i]
		posSeq := dataContainer.SamplingPosSeqs[i]
		pyhsmm.addWordSeqAsCustomer(wordSeq, posSeq)
	}
	bar.Finish()
}

func adjustGFeatsSlice(pyhsmm *PYHSMM, gFeatsSlice []GenerativeFeatures, apiParam APIParam) []GenerativeFeatures {
	maxSentLen := -1
	for _, gFeats := range gFeatsSlice {
		if maxSentLen < len(gFeats) {
			maxSentLen = len(gFeats)
		}
	}

	ch := make(chan int, apiParam.ThreadsNum)
	wg := sync.WaitGroup{}

	adjustGFeatsSlice := make([]GenerativeFeatures, len(gFeatsSlice), len(gFeatsSlice))
	for b := 0; b < len(gFeatsSlice); b++ {
		ch <- 1
		wg.Add(1)
		go func(b int) {
			adjustGFeatsSlice[b] = make(GenerativeFeatures, maxSentLen, maxSentLen)
			for t := 0; t < maxSentLen; t++ {
				adjustGFeatsSlice[b][t] = make([][][][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
				for k := 0; k < pyhsmm.maxWordLength; k++ {
					adjustGFeatsSlice[b][t][k] = make([][][]float64, pyhsmm.PosSize+2, pyhsmm.PosSize+2)
					for z := 0; z < pyhsmm.PosSize+2; z++ {
						adjustGFeatsSlice[b][t][k][z] = make([][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
						for j := 0; j < pyhsmm.maxWordLength; j++ {
							adjustGFeatsSlice[b][t][k][z][j] = make([]float64, pyhsmm.PosSize+2, pyhsmm.PosSize+2)
							for r := 0; r < pyhsmm.PosSize+2; r++ {
								adjustGFeatsSlice[b][t][k][z][j][r] = apiParam.LowerBound
							}
						}
					}
				}
			}
			<-ch
			wg.Done()
		}(b)
	}
	wg.Wait()

	for b := range gFeatsSlice {
		ch <- 1
		wg.Add(1)
		go func(b int) {
			for t, gFeat := range gFeatsSlice[b] {
				for k, v := range gFeat {
					for z, vv := range v {
						for j, vvv := range vv {
							for r, vvvv := range vvv {
								adjustGFeatsSlice[b][t][k][z][j][r] = vvvv
							}
						}
					}
				}
			}
			<-ch
			wg.Done()
		}(b)
	}
	wg.Wait()

	// for b, gFeats := range gFeatsSlice {
	// 	adjustGFeatsSlice[b] = gFeats
	// 	for t, gFeat := range gFeats {
	// 		adjustGFeatsSlice[b][t] = gFeat
	// 	}
	// }
	return adjustGFeatsSlice
}

func (pyhsmm *PYHSMM) getGenerativeFeatures(sent []string, lowerBound float64) GenerativeFeatures {

	// initialize gFeats[t][k][z][j][r]
	gFeats := make(GenerativeFeatures, len(sent)+1, len(sent)+1)
	for t := 0; t < len(sent)+1; t++ {
		gFeats[t] = make([][][][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			gFeats[t][k] = make([][][]float64, pyhsmm.PosSize+2, pyhsmm.PosSize+2)
			for z := 0; z < pyhsmm.PosSize+2; z++ {
				gFeats[t][k][z] = make([][]float64, pyhsmm.maxWordLength, pyhsmm.maxWordLength)
				for j := 0; j < pyhsmm.maxWordLength; j++ {
					gFeats[t][k][z][j] = make([]float64, pyhsmm.PosSize+2, pyhsmm.PosSize+2)
					for r := 0; r < pyhsmm.PosSize+2; r++ {
						gFeats[t][k][z][j][r] = lowerBound
					}
				}
			}
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
					posScoreLog := eachScoreForPos[pos][pyhsmm.bosPos]
					score := wordScoreLog + posScoreLog
					if math.IsNaN(score) {
						errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v)", wordScoreLog, posScoreLog)
						panic(errMsg)
					}
					j := 0
					r := pyhsmm.bosPos
					if score < lowerBound {
						score = lowerBound
					}
					gFeats[t][k][pos][j][r] = score
					continue
				}
				// gFeats[t][k][pos] = 0.0
				for j := 0; j < pyhsmm.maxWordLength; j++ {
					if t-k-(j+1) >= 0 {
						//
					} else {
						// for r := 0; r < pyhsmm.PosSize+2; r++ {
						// 	gFeats[t][k][pos][j][r] = 1.0
						// }
						continue
					}
					for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
						wordScoreLog := eachScoreForWord[t][k][pos][j]
						posScoreLog := eachScoreForPos[pos][prevPos]
						score := wordScoreLog + posScoreLog
						if math.IsNaN(score) {
							errMsg := fmt.Sprintf("forward error! score is NaN. wordScore (%v), posScore, (%v)", wordScoreLog, posScoreLog)
							panic(errMsg)
						}
						if score < lowerBound {
							score = lowerBound
						}
						gFeats[t][k][pos][j][prevPos] = score
					}
				}
			}
		}
	}

	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	uPos := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := pyhsmm.npylms[0].vpylm.hpylm.Base

	t := len(sent)
	k := 0
	for j := 0; j < pyhsmm.maxWordLength; j++ {
		for prevPos := 0; prevPos < pyhsmm.PosSize; prevPos++ {
			if t-k-(j+1) >= 0 {
				u[0] = strings.Join(sent[(t-k-(j+1)):(t-k)], pyhsmm.npylms[0].splitter)
				wordScore, _ := pyhsmm.npylms[pyhsmm.eosPos].CalcProb(pyhsmm.eos, u, base)
				uPos[0] = strconv.Itoa(prevPos)
				posScore, _ := pyhsmm.posHpylm.CalcProb(strconv.Itoa(pyhsmm.eosPos), uPos, pyhsmm.posHpylm.Base)
				score := math.Log(wordScore) + math.Log(posScore)
				if score < lowerBound {
					score = lowerBound
				}
				gFeats[t][k][pyhsmm.eosPos][j][prevPos] = score
			}
		}
	}

	return gFeats
}

func (pyhsmm *PYHSMM) backwardJESSCM(sent []string, forwardScore forwardScoreForWordAndPosType, sampling bool, lambda0 float64, discScore [][][]float64, discScoreT [][]float64, lowerBound float64) (context, []int) {
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
					gScore := math.Log(wordScore) + math.Log(posScore)
					if gScore < lowerBound {
						gScore = lowerBound
					}
					score := 0.0
					if prevPos != pyhsmm.eosPos {
						score = (lambda0 * gScore) + discScore[t][k][prevPos] + discScoreT[nextPos][prevPos] + forwardScore[t-(k+1)][j][nextPos]
					} else {
						score = (lambda0 * gScore) + discScoreT[nextPos][prevPos] + forwardScore[t-(k+1)][j][nextPos]
					}
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
