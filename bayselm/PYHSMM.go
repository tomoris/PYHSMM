package bayselm

import (
	"math"
	"math/rand"
	"sync"

	"github.com/cheggaaa/pb/v3"
)

type forwardScoreForWordAndPosType [][][]float64

// PYHSMM contains posSize-th NPYLM instances.
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

// NewPYHSMM returns PYHSMM instance.
func NewPYHSMM(initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, PosSize int) *PYHSMM {

	npylms := make([]*NPYLM, PosSize+1, PosSize+1)
	for pos := 0; pos < PosSize+1; pos++ {
		npylms[pos] = NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength)
	}

	pyhsmm := &PYHSMM{npylms, maxNgram, maxWordLength, bos, "<EOS>", "<BOW>", "<EOW>", PosSize, PosSize}

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
	for pos := 0; pos < pyhsmm.PosSize; pos++ {
		pyhsmm.npylms[pos].poissonCorrection()
		pyhsmm.npylms[pos].estimateHyperPrameters()
		pyhsmm.npylms[pos].vpylm.hpylm.estimateHyperPrameters()
	}
	return
}

// TestWordSegmentationAndPOSTagging inferences word segmentation from input unsegmented texts.
func (pyhsmm *PYHSMM) TestWordSegmentationAndPOSTagging(sents [][]rune, threadsNum int) ([][]string, [][]int) {
	wordSeqs := make([][]string, len(sents), len(sents))
	posSeqs := make([][]int, len(sents), len(sents))
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
	base := float64(0.0)
	for t := 0; t < len(sent); t++ {
		for k := 0; k < pyhsmm.maxWordLength; k++ {
			for pos := 0; pos < pyhsmm.PosSize; pos++ {
				if t-k >= 0 {
					word = string(sent[(t - k) : t+1])
					u[0] = pyhsmm.bos
					base = pyhsmm.npylms[pos].calcBase(word)
					if t-k == 0 {
						score, _ := pyhsmm.npylms[pos].CalcProb(word, u, base)
						forwardScore[t][k][pos] = float64(math.Log(float64(score)))
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
							score = float64(math.Log(float64(score)) + float64(forwardScore[t-(k+1)][j][prevPos]))
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
		scoreArray := make([]float64, pyhsmm.maxWordLength*pyhsmm.PosSize, pyhsmm.maxWordLength*pyhsmm.PosSize)
		maxScore := float64(math.Inf(-1))
		maxJ := -1
		maxNextPos := -1
		sumScore := float64(0.0)
		for j := 0; j < pyhsmm.maxWordLength; j++ {
			for nextPos := 0; nextPos < pyhsmm.PosSize; nextPos++ {
				if t-k-(j+1) >= 0 {
					u[0] = string(sent[(t - k - (j + 1)):(t - k)])
					score, _ := pyhsmm.npylms[prevPos].CalcProb(prevWord, u, base)
					score = float64(math.Log(float64(score)) + float64(forwardScore[t-(k+1)][j][nextPos]))
					if score > maxScore {
						maxScore = score
						maxJ = j
						maxNextPos = nextPos
					}
					score = float64(math.Exp(float64(score)))
					scoreArray[j*pyhsmm.PosSize+nextPos] = score
					sumScore += score
				} else {
					scoreArray[j*pyhsmm.PosSize+nextPos] = float64(math.Inf(-1))
				}
			}
		}
		j := 0
		nextPos := 0
		if sampling {
			r := float64(rand.Float64()) * sumScore
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

func (pyhsmm *PYHSMM) addWordSeqAsCustomer(wordSeq context, posSeq []int) {
	u := make(context, pyhsmm.maxNgram-1, pyhsmm.maxNgram-1)
	base := float64(0.0)
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

func (pyhsmm *PYHSMM) removeWordSeqAsCustomer(wordSeq context, posSeq []int) {
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
