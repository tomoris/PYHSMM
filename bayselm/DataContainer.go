package bayselm

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// DataContainer contains information of sentences, word sequences and their part-of-speech sequence.
type DataContainer struct {
	Sents                 [][]string
	SamplingWordSeqs      []context
	SamplingPosSeqs       [][]int // for PYHSMM
	SamplingDepthMemories [][]int // for VPYLM
	Size                  int
}

// // NewDataContainerFromSents returns DataContainer instance.
// // input file is required unsegmented texts (not split space)
// func NewDataContainerFromSents(sents []string) *DataContainer {
// 	dataContainer := new(DataContainer)

// 	count := 0
// 	for _, sentStr := range sents {
// 		loweredStringSent := strings.ToLower(sentStr)
// 		sent := []rune(loweredStringSent)
// 		if len(sent) > 0 {
// 			dataContainer.Sents = append(dataContainer.Sents, sent)
// 			dataContainer.SamplingWordSeqs = append(dataContainer.SamplingWordSeqs, make(context, 0, len(sent)))
// 			dataContainer.SamplingPosSeqs = append(dataContainer.SamplingPosSeqs, make([]int, 0, len(sent)))
// 			dataContainer.SamplingDepthMemories = append(dataContainer.SamplingDepthMemories, make([]int, 0, len(sent)))
// 			count++
// 		}
// 	}
// 	dataContainer.Size = count
// 	return dataContainer
// }

// NewDataContainer returns DataContainer instance.
// input file is required unsegmented texts (not split space)
func NewDataContainer(filePath string, splitter string) *DataContainer {
	dataContainer := new(DataContainer)

	f, ok := os.Open(filePath)
	if ok != nil {
		errMsg := fmt.Sprintf("cannot open filePath (%v)", filePath)
		panic(errMsg)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	count := 0
	for sc.Scan() {
		if ok := sc.Err(); ok != nil {
			errMsg := fmt.Sprintf("read error in filePath (%v): line %v", filePath, count)
			panic(errMsg)
		}

		loweredStringSent := strings.ToLower(sc.Text())
		sent := strings.Split(loweredStringSent, splitter)
		if len(sent) > 0 {
			dataContainer.Sents = append(dataContainer.Sents, sent)
			dataContainer.SamplingWordSeqs = append(dataContainer.SamplingWordSeqs, make(context, 0, len(sent)))
			dataContainer.SamplingPosSeqs = append(dataContainer.SamplingPosSeqs, make([]int, 0, len(sent)))
			dataContainer.SamplingDepthMemories = append(dataContainer.SamplingDepthMemories, make([]int, 0, len(sent)))
			count++
		}
	}
	dataContainer.Size = count
	return dataContainer
}

// NewDataContainerFromAnnotatedData returns DataContainer instance.
// input file is required segmented texts (split space)
func NewDataContainerFromAnnotatedData(filePath string) *DataContainer {
	dataContainer := new(DataContainer)

	f, ok := os.Open(filePath)
	if ok != nil {
		errMsg := fmt.Sprintf("cannot open filePath (%v)", filePath)
		panic(errMsg)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	count := 0
	for sc.Scan() {
		if ok := sc.Err(); ok != nil {
			errMsg := fmt.Sprintf("read error in filePath (%v): line %v", filePath, count)
			panic(errMsg)
		}

		sentStr := sc.Text()
		sent := strings.Split(sentStr, "")
		if len(sent) > 0 {
			dataContainer.Sents = append(dataContainer.Sents, sent)
			wordSeq := make(context, 0, len(sent))
			wordSeq = context(strings.Split(sentStr, " "))
			if len(wordSeq) == 0 {
				continue
			}
			dataContainer.SamplingWordSeqs = append(dataContainer.SamplingWordSeqs, wordSeq)
			dataContainer.SamplingPosSeqs = append(dataContainer.SamplingPosSeqs, make([]int, len(wordSeq), len(wordSeq)))
			dataContainer.SamplingDepthMemories = append(dataContainer.SamplingDepthMemories, make([]int, 0, len(wordSeq)))
			count++
		}
	}
	dataContainer.Size = count
	return dataContainer
}

// GetWordSeq returns i-th wordSeq ([]string) for python binding.
func (dataContainer *DataContainer) GetWordSeq(i int) []string {
	return dataContainer.SamplingWordSeqs[i]
}

// GetSentString returns i-th sent (string) for python binding.
// e.g., sent = "this is an example of sent"
func (dataContainer *DataContainer) GetSentString(i int) string {
	if i > dataContainer.Size {
		errMsg := fmt.Sprintf("GetSentString error. index i (%v) is bigger than size (%v)", i, dataContainer.Size)
		panic(errMsg)
	}
	return strings.Join((dataContainer.Sents[i]), "")
}
