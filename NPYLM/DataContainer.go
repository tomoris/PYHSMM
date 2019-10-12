package NPYLM

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

type DataContainer struct {
	Sents            [][]rune
	SamplingWordSeqs []context
	SamplingPosSeqs  [][]int
	Size             int
}

func NewDataContainer(filePath string) *DataContainer {
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
		sent := []rune(loweredStringSent)
		if len(sent) > 0 {
			dataContainer.Sents = append(dataContainer.Sents, sent)
			dataContainer.SamplingWordSeqs = append(dataContainer.SamplingWordSeqs, make(context, 0, len(sent)))
			dataContainer.SamplingPosSeqs = append(dataContainer.SamplingPosSeqs, make([]int, 0, len(sent)))
			count++
		}
	}
	dataContainer.Size = count
	return dataContainer
}

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
		sent := []rune(strings.Replace(sentStr, " ", "", -1))
		if len(sent) > 0 {
			dataContainer.Sents = append(dataContainer.Sents, sent)
			wordSeq := make(context, 0, len(sent))
			wordSeq = context(strings.Fields(sentStr))
			dataContainer.SamplingWordSeqs = append(dataContainer.SamplingWordSeqs, wordSeq)
			count++
		}
	}
	dataContainer.Size = count
	return dataContainer
}
