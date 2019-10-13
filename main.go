package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"time"

	"github.com/tomoris/PYHSMM/bayselm"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	var (
		flagRawFilePath = flag.String("rawTexts", "", "training file path")
		// flagSegmentedFilePath = flag.String("segmentedTexts", "alice.txt", "training file path")
		flagEpoch = flag.Int("epoch", 1, "epoch size")

		flagInitialTheta  = flag.Float64("theta", 2.0, "initial hyper-parameter in NPYLM")
		flagInitialD      = flag.Float64("d", 0.1, "initial hyper-parameter in NPYLM")
		flagGammaA        = flag.Float64("gammaA", 1.0, "hyper-parameter in NPYLM")
		flagGammaB        = flag.Float64("gammaB", 1.0, "hyper-parameter in NPYLM")
		flagBetaA         = flag.Float64("betaA", 1.0, "hyper-parameter in NPYLM")
		flagBetaB         = flag.Float64("betaB", 1.0, "hyper-parameter in NPYLM")
		flagAlpha         = flag.Float64("alpha", 1.0, "hyper-parameter in NPYLM")
		flagBeta          = flag.Float64("beta", 1.0, "hyper-parameter in NPYLM")
		flagMaxNgram      = flag.Int("ngram", 2, "number of maximum n-gram")
		flagMaxWordLength = flag.Int("length", 15, "number of maximum word length")
		flagThreads       = flag.Int("threads", 8, "number of threads")
		flagBatch         = flag.Int("batch", 32, "batch size")
	)
	flag.Parse()

	runtime.GOMAXPROCS(*flagThreads)
	fmt.Println("Building model")
	npylm := bayselm.NewNPYLM(*flagInitialTheta, *flagInitialD, *flagGammaA, *flagGammaB, *flagBetaA, *flagBetaB, *flagAlpha, *flagBeta, *flagMaxNgram, *flagMaxWordLength)
	// npylm := NPYLM.NewPYHSMM(*flagInitialTheta, *flagInitialD, *flagGammaA, *flagGammaB, *flagBetaA, *flagBetaB, *flagAlpha, *flagBeta, *flagMaxNgram, *flagMaxWordLength, 10)
	fmt.Println("Loading data and initialize model")
	dataContainer := bayselm.NewDataContainer(*flagRawFilePath)
	npylm.Initialize(dataContainer.Sents, dataContainer.SamplingWordSeqs)
	// dataContainer := NPYLM.NewDataContainerFromAnnotatedData(*flagSegmentedFilePath)
	// npylm.InitializeFromAnnotatedData(dataContainer.Sents, dataContainer.SamplingWordSeqs)
	// npylm.Initialize(dataContainer.Sents, dataContainer.SamplingWordSeqs, dataContainer.SamplingPosSeqs)
	// dataContainer := NPYLM.NewDataContainerFromAnnotatedData(*flagFilePath)
	// for i := 0; i < dataContainer.Size; i++ {
	// 	for j, word := range dataContainer.SamplingWordSeqs[i] {
	// 		runeWord := []rune(word)
	// 		if len(runeWord) >= *flagMaxWordLength {
	// 			wordSegment := make([]string, 0, len(runeWord))
	// 			for _, char := range runeWord {
	// 				wordSegment = append(wordSegment, string(char))
	// 			}
	// 			tmpSamplingWordSeq := dataContainer.SamplingWordSeqs[i][j+1:]
	// 			dataContainer.SamplingWordSeqs[i] = append(dataContainer.SamplingWordSeqs[i][:j], wordSegment...)
	// 			dataContainer.SamplingWordSeqs[i] = append(dataContainer.SamplingWordSeqs[i], tmpSamplingWordSeq...)
	// 			fmt.Println("adjust", dataContainer.SamplingWordSeqs[i])
	// 		}
	// 	}
	// 	npylm.AddWordSeqAsCustomer(dataContainer.SamplingWordSeqs[i])
	// }
	// for epoch := 0; epoch < 10; epoch++ {
	// 	npylm.AddWordSeqAsCustomer([]string{"これ", "は", "ペン", "です", "。"})
	// }

	// for i := 0 ; i < dataContainer.Size; i ++ {
	// 	sent := dataContainer.Sents[i]
	// 	fmt.Println(strings.Join(npylm.Test(sent), " "))
	// }
	// defer profile.Start(profile.ProfilePath(".")).Stop()
	fmt.Println("Training model")
	for epoch := 0; epoch < *flagEpoch; epoch++ {
		// fmt.Println("prev", dataContainer.SamplingWordSeqs[0])
		npylm.TrainWordSegmentation(dataContainer, *flagThreads, *flagBatch)
		testSize := 50
		wordSeqs := npylm.TestWordSegmentation(dataContainer.Sents[:testSize], *flagThreads)
		for i := 0; i < testSize; i++ {
			fmt.Println("test", wordSeqs[i])
		}
	}

}
