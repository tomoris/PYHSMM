package main

import (
	"C"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/tomoris/PYHSMM/bayselm"
	"gopkg.in/alecthomas/kingpin.v2"

	"net/http"

	"github.com/gin-gonic/gin"
)

var (
	args = kingpin.New("bayselm", "Baysian n-gram language model.")

	lm                 = args.Command("lm", "training language model from segmented texts")
	modelForLM         = lm.Flag("model", "n-gram model").Required().Enum("ngram", "hpylm", "vpylm", "npylm", "pyhsmm")
	trainFilePathForLM = lm.Flag("trainFile", "training file path. the texts are segmented space.").Required().String()
	testFilePathForLM  = lm.Flag("testFile", "test file path. the texts are segmented space.").Required().String()

	ws                 = args.Command("ws", "training word segmentation from unsegmented texts")
	modelForWS         = ws.Flag("model", "unsupervised word segmentation model").Required().Enum("npylm", "pyhsmm")
	trainFilePathForWS = ws.Flag("trainFile", "training file path. the texts are unsegmented.").Required().String()
	testFilePathForWS  = ws.Flag("testFile", "test file path. the texts are unsegmented.").Default("").String()

	wsTest                = args.Command("wsTest", "training word segmentation from unsegmented texts")
	modelForWSTest        = wsTest.Flag("model", "unsupervised word segmentation model").Required().Enum("npylm", "pyhsmm")
	testFilePathForWSTest = wsTest.Flag("testFile", "test file path. the texts are unsegmented.").Default("").String()
	loadFile              = wsTest.Flag("loadFile", "file path to load model").String()

	api                        = args.Command("api", "launch API for intergrating PYHSMM and discriminative model (semi-Markov CRF)")
	trainFilePathForAPI        = api.Flag("trainFile", "training file path. the texts are unsegmented.").Required().String()
	trainGeneralFilePathForAPI = api.Flag("trainGeneralFilePathForAPI", "training file path. the texts are unsegmented.").Required().String()
	oLabelID                   = api.Flag("oLabelID", "o label id").Required().Int()

	randSeed      = args.Flag("randSeed", "random seed").Default("0").Int64()
	maxSentLen    = args.Flag("maxSentLen", "maxSentLen").Default("128").Int()
	maxNgram      = args.Flag("maxNgram", "hyper-parameter in HPYLM - PYHSMM").Default("2").Int()
	initialTheta  = args.Flag("theta", "initial hyper-parameter in HPYLM - PYHSMM").Default("2.0").Float64()
	initialD      = args.Flag("d", "initial hyper-parameter in HPYLM - PYHSMM").Default("0.9").Float64()
	gammaA        = args.Flag("gammaA", "hyper-parameter in HPYLM - PYHSMM").Default("1.0").Float64()
	gammaB        = args.Flag("gammaB", "hyper-parameter in HPYLM - PYHSMM").Default("1.0").Float64()
	betaA         = args.Flag("betaA", "hyper-parameter in HPYLM - PYHSMM").Default("1.0").Float64()
	betaB         = args.Flag("betaB", "hyper-parameter in HPYLM - PYHSMM").Default("1.0").Float64()
	vocabSize     = args.Flag("vocabSize", "hyper-parameter in HPYLM - VPYLM (default parameter is size of character vocab. (utf-8))").Default("2097152.0").Float64()
	alpha         = args.Flag("alpha", "hyper-parameter in VPYLM - PYHSMM (used sampling depth in VPYLM)").Default("1.0").Float64()
	beta          = args.Flag("beta", "hyper-parameter in VPYLM - PYHSMM (used sampling depth in VPYLM)").Default("1.0").Float64()
	maxWordLength = args.Flag("maxWordLength", "hyper-parameter in NPYLM - PYHSMM").Default("10").Int()
	posSize       = args.Flag("posSize", "hyper-parameter in NPYLM - PYHSMM").Default("10").Int()
	epoch         = args.Flag("epoch", "hyper-parameter in HPYLM - PYHSMM").Default("100").Int()
	batch         = args.Flag("batch", "hyper-parameter in NPYLM - PYHSMM").Default("16").Int()
	threads       = args.Flag("threads", "hyper-parameter in NPYLM - PYHSMM").Default("8").Int()
	splitter      = args.Flag("splitter", "hyper-parameter in NPYLM - PYHSMM").Default("").String()

	saveFile   = args.Flag("saveFile", "file path to save model").String()
	saveFormat = args.Flag("saveFormat", "model save format").Default("notindent").Enum("notindent", "indent")
)

func trainLanguageModel() {
	model, ok := bayselm.GenerateNgramLM(*modelForLM, *initialTheta, *initialD, *gammaA, *gammaB, *betaA, *betaB, *alpha, *beta, *maxNgram, *maxWordLength, *posSize, 1.0 / *vocabSize)
	if !ok {
		panic("Building model error")
	}
	dataContainerForTrain := bayselm.NewDataContainerFromAnnotatedData(*trainFilePathForLM)
	dataContainerForTest := bayselm.NewDataContainerFromAnnotatedData(*testFilePathForLM)
	time.Sleep(3)
	for e := 0; e < *epoch; e++ {
		model.Train(dataContainerForTrain)
		perplexity := bayselm.CalcPerplexity(model, dataContainerForTest)
		fmt.Println("Perplexity = ", perplexity)
	}
	if *saveFile != "" {
		bayselm.Save(model.(bayselm.NgramLM), *saveFile, *saveFormat)
		// セーブしたものと同じモデルをロードできるかの確認
		// loadModel := bayselm.Load(*modelForLM, *saveFile)
		// perplexity := bayselm.CalcPerplexity(loadModel, dataContainerForTest)
		// fmt.Println("Perplexity = ", perplexity)
	}
	return
}

func trainWordSegmentation(modelForWS string, trainFilePathForWS string, testFilePathForWS string, initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, posSize int, base float64, epoch int, threads int, batch int, saveFile string, saveFormat string, splitter string, maxSentLen int) {
	runtime.GOMAXPROCS(threads)
	model, ok := bayselm.GenerateUnsupervisedWSM(modelForWS, initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, posSize, base, splitter)
	if !ok {
		panic("Building model error")
	}
	dataContainer := bayselm.NewDataContainer(trainFilePathForWS, splitter, maxSentLen)
	// dataContainer := bayselm.NewDataContainerFromAnnotatedData(trainFilePathForWS)
	if testFilePathForWS == "" {
		testFilePathForWS = trainFilePathForWS
	}
	model.Initialize(dataContainer)
	// model.InitializeFromAnnotatedData(dataContainer)
	dataContainerForTest := bayselm.NewDataContainer(testFilePathForWS, splitter, maxSentLen)
	for e := 0; e < epoch; e++ {
		model.TrainWordSegmentation(dataContainer, threads, batch)
		testSize := dataContainerForTest.Size
		wordSeqs := model.TestWordSegmentation(dataContainerForTest.Sents[:testSize], threads)
		for i := 0; i < testSize; i++ {
			if splitter == "" {
				fmt.Println(e, "test", wordSeqs[i])
			} else {
				fmt.Println(e, "test", strings.Join(wordSeqs[i], "_"))
			}
		}
		scoreDivWordSize, scoreDivSentSize := model.CalcTestScore(wordSeqs, threads)
		fmt.Println("scoreDivWordSize = ", scoreDivWordSize, "\t", "scoreDivSentSize = ", scoreDivSentSize)
		model.ShowParameters()
	}
	if saveFile != "" {
		bayselm.Save(model.(bayselm.NgramLM), saveFile, saveFormat)
		// セーブしたものと同じモデルをロードできるかの確認
		// var loadModel bayselm.UnsupervisedWSM = bayselm.Load(modelForWS, saveFile).(bayselm.UnsupervisedWSM)
		// testSize := 10
		// wordSeqs := loadModel.TestWordSegmentation(dataContainer.Sents[:testSize], threads)
		// for i := 0; i < testSize; i++ {
		// 	fmt.Println("test", wordSeqs[i])
		// }
		// for e := 0; e < epoch; e++ {
		// 	model.TrainWordSegmentation(dataContainer, threads, batch)
		// 	testSize := 10
		// 	wordSeqs := model.TestWordSegmentation(dataContainer.Sents[:testSize], threads)
		// 	for i := 0; i < testSize; i++ {
		// 		fmt.Println("test", wordSeqs[i])
		// 	}
		// }
	}
	return
}

func testWordSegmentation(modelForWS string, testFilePathForWS string, loadFile string, threads int, splitter string, maxSentLen int) {
	var model bayselm.UnsupervisedWSM = bayselm.Load(modelForWS, loadFile).(bayselm.UnsupervisedWSM)
	dataContainerForTest := bayselm.NewDataContainer(testFilePathForWS, splitter, maxSentLen)
	testSize := dataContainerForTest.Size
	wordSeqs := model.TestWordSegmentation(dataContainerForTest.Sents[:testSize], threads)
	for i := 0; i < testSize; i++ {
		var newline string
		for _, token := range wordSeqs[i] {
			newline += token + " "
		}
		fmt.Println(newline[:len(newline)-1])
	}
}

func launchAPI(trainFilePathForAPI string, trainGeneralFilePathForAPI string, initialTheta float64, initialD float64, gammaA float64, gammaB float64, betaA float64, betaB float64, alpha float64, beta float64, maxNgram int, maxWordLength int, posSize int, base float64, splitter string, threads int, oLabelID int, maxSentLen int) {
	runtime.GOMAXPROCS(threads)
	model := bayselm.NewPYHSMM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta, maxNgram, maxWordLength, posSize, splitter)
	dataContainer := bayselm.NewDataContainer(trainFilePathForAPI, splitter, maxSentLen)
	dataContainerGeneralDomain := bayselm.NewDataContainer(trainGeneralFilePathForAPI, splitter, maxSentLen)

	engine := gin.Default()
	engine.GET("/GetPYHSMMFeatsAPI", func(c *gin.Context) {
		var apiParam bayselm.APIParam
		if err := c.BindJSON(&apiParam); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"status": "BadRequest"})
			return
		}
		gfeatsSlice := bayselm.GetPYHSMMFeatsAPI(model, dataContainer, apiParam)
		c.JSON(http.StatusOK, gin.H{"gFeatsSlice": gfeatsSlice})
	})
	engine.GET("/GetPYHSMMFeatsFromSentsAPI", func(c *gin.Context) {
		var apiParam bayselm.APIParam
		if err := c.BindJSON(&apiParam); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"status": "BadRequest"})
			return
		}
		gfeatsSlice := bayselm.GetPYHSMMFeatsFromSentsAPI(model, dataContainer, apiParam)
		c.JSON(http.StatusOK, gin.H{"gFeatsSlice": gfeatsSlice})
	})
	engine.POST("/AddCustomerUsingForwardScoreAPI", func(c *gin.Context) {
		var apiParam bayselm.APIParam
		if err := c.BindJSON(&apiParam); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"status": "BadRequest"})
			return
		}
		bayselm.AddCustomerUsingForwardScoreAPI(model, dataContainer, apiParam)
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})
	engine.DELETE("/RemoveCustomerAPI", func(c *gin.Context) {
		var apiParam bayselm.APIParam
		if err := c.BindJSON(&apiParam); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"status": "BadRequest"})
			return
		}
		bayselm.RemoveCustomerAPI(model, dataContainer, apiParam)
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})
	engine.POST("/TrainGeneralDomainAPI", func(c *gin.Context) {
		bayselm.TrainFromAnnotatedCorpus(model, dataContainerGeneralDomain)
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})
	engine.POST("/InitializeAPI", func(c *gin.Context) {
		model.Initialize(dataContainer)
		for i, sent := range dataContainerGeneralDomain.Sents {
			dataContainerGeneralDomain.SamplingWordSeqs[i] = sent
			dataContainerGeneralDomain.SamplingPosSeqs[i] = make([]int, len(sent), len(sent))
			for j := range dataContainerGeneralDomain.SamplingPosSeqs[i] {
				dataContainerGeneralDomain.SamplingPosSeqs[i][j] = oLabelID
			}
		}
		bayselm.AddWordSeqAsCustomerAPI(model, dataContainerGeneralDomain)
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})
	engine.Run(":3000")
}

func main() {
	rand.Seed(0)
	switch kingpin.MustParse(args.Parse(os.Args[1:])) {
	case lm.FullCommand():
		rand.Seed(*randSeed)
		trainLanguageModel()
	case ws.FullCommand():
		rand.Seed(*randSeed)
		trainWordSegmentation(*modelForWS, *trainFilePathForWS, *testFilePathForWS, *initialTheta, *initialD, *gammaA, *gammaB, *betaA, *betaB, *alpha, *beta, *maxNgram, *maxWordLength, *posSize, 1.0 / *vocabSize, *epoch, *threads, *batch, *saveFile, *saveFormat, *splitter, *maxSentLen)
	case wsTest.FullCommand():
		rand.Seed(*randSeed)
		testWordSegmentation(*modelForWSTest, *testFilePathForWSTest, *loadFile, *threads, *splitter, *maxSentLen)
	case api.FullCommand():
		rand.Seed(*randSeed)
		launchAPI(*trainFilePathForAPI, *trainGeneralFilePathForAPI, *initialTheta, *initialD, *gammaA, *gammaB, *betaA, *betaB, *alpha, *beta, *maxNgram, *maxWordLength, *posSize, 1.0 / *vocabSize, *splitter, *threads, *oLabelID, *maxSentLen)
	}
	return
}
