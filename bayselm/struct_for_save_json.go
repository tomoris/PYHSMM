package bayselm

import "gonum.org/v1/gonum/stat/distuv"

type restaurantJSON struct {
	Tables map[string][]table // word to tables
	// simple p(word|context) = customerCount[word] / totalCustomerCount
	CustomerCount              map[string]newUint // word to count int the restaurant.
	TotalCustomerCount         newUint
	TotalTableCountForCustomer map[string]newUint // word to number of word serving tables
	TotalTableCount            newUint            // number of tables

	Stop newUint // number of stop for stop probability in n-gram
	Pass newUint // number of pass for stop probability in n-gram
}

type HPYLMJSON struct {
	Restaurants map[string]*restaurantJSON // context to restaurant

	MaxDepth int
	Theta    []float64 // parameters for Pitman-Yor process
	D        []float64 // parameters for Pitman-Yor process
	GammaA   []float64 // hyper-parameters using gamma distributionfor to estimate theta
	GammaB   []float64 // hyper-parameters using gamma distributionfor to estimate theta
	BetaA    []float64 // hyper-parameters using beta distributionfor to estimate d
	BetaB    []float64 // hyper-parameters using beta distributionfor to estimate d
	Base     float64
}

type VPYLMJSON struct {
	Hpylm *HPYLMJSON
	Alpha float64 // hyper-parameter for beta distribution to estimate stop probability
	Beta  float64 // hyper-parameter for beta distribution to estimate stop probability
}

type NPYLMJSON struct {
	*HPYLMJSON
	// add
	Vpylm *VPYLMJSON

	MaxNgram      int
	MaxWordLength int
	Bos           string
	Eos           string
	Bow           string
	Eow           string

	Poisson     distuv.Poisson
	Length2prob []float64

	Word2sampledDepthMemory map[string][][]int
}

type PYHSMMJSON struct {
	Npylms   []*NPYLMJSON
	PosHpylm interface{}

	MaxNgram      int
	MaxWordLength int
	Bos           string
	Eos           string
	Bow           string
	Eow           string

	PosSize int
	EosPos  int
	BosPos  int
}
