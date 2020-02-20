# PYHSMM implemented in Go
Implementation of Pitman-Yor Hidden Semi-Markov Model (PYSHMM) (Uchiumi et al., 2015) implemented in golang. PYHSMM is a non-parametric Bayesian model for inducing words and part-of-speech tags from unsegmented texts without annotated data.

This repository also contain other Bayesian n-gram based language models such as HPYLM, and NPYLM.

## Prerequisites
- Go: 12.5  
    - gonum.org/v1/gonum/stat/distuv  
    - github.com/cheggaaa/pb/v3  
    - gopkg.in/alecthomas/kingpin.v2  
    - github.com/go-python/gopy  

- Python: 3 (we test the program in python 3.6)  


## Installing
```
go get github.com/tomoris/PYHSMM
cd $GOPATH/src/github.com/tomoris/PYHSMM
go build main.go
```

### for Python Extention (if you want)
```
gopy build -vm=`which python3` -output=pylib github.com/tomoris/PYHSMM/bayselm
```

## Usage
Training language model.  
`./main lm --model hpylm --maxNgram 2 --trainFile data/sample.train.word.txt --testFile data/sample.test.word.txt`  
Training word segmentation model without labeled data.  
`./main ws --model npylm --maxNgram 2 --trainFile data/sample.txt --threads 8 --saveFile sample.model.json`  

### for Python Extention (if you want)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./pylib
export PYTHONPATH=$PYTHONPATH:./pylib
python main.py --mode ws --model pyhsmm --trainFile data/sample.txt --posSize 5 --epoch 1 --saveFile model.json
```



### Models
```
ngram: Interporated n-gram model
hpylm: Hierarchical Pitman-Yor Language Model
vpylm: Variable order HPYLM
npylm: Nested Pitman-Yor Language Model (unsupervised word segmentation model)
pyhsmm: Pitman-Yor Hidden Semi-Markov Model (unsupervised word segmentation and POS induction model)
```

## Evaluation

### Language Model
We evaluated the language models of them on BCCWJ. The definition of words is super short word unit. Size of training sentences is 57,281. Size of test sentences is 3,024. We note that npylm and pyhsmm were smoothed by charcter level language model. It means unfair comparison but we show the results.

| Model  | perplexty |    n | number of POS tags |
| :----: | --------: | ---: | -----------------: |
| ngram  |     409.9 |    3 |                  - |
| hpylm  |     130.4 |    3 |                  - |
| vpylm  |     131.3 |    8 |                  - |
| npylm  |     199.9 |    2 |                  - |
| pyhsmm |     200.1 |    2 |                  1 |
| pyhsmm |     234.3 |    2 |                  5 |

### Unsupervised Word Segmentation

| Model  | Precision | Recall | F-score |
| :----: | --------: | -----: | ------: |
| npylm  |         ? |      ? |       ? |
| pyhsmm |         ? |      ? |       ? |


Example of "Alice in Wonderland".  
 - hyperparameters
   - model: npylm
   - epoch: 100
   - maxWordLength: 10
   - alpha: 4.0
   - others: default parameters
> 99 test [alice was beginn ing toget very tired of sitting by hersister onthe]  
> 99 test [bank ,and ofhaving nothing todo : once ortwice shehad peeped intothe]  
> 99 test [book hersister was reading ,but ithad no pictures or c onv ersation sin]  
> 99 test [it ,‘ and what is theuseof a book ,’ thought alice ‘ without pictures or]  
> 99 test [c onv ersation s ?’]  
> 99 test [soshe was consider ing inher own mind ( aswe ll asshecould ,for the]  
> 99 test [hot day madeher feel very sleepy ands tupid ), whether the pleasure]  
> 99 test [of making ada isy-chain wouldbe worth the trouble of g etting up and]  
> 99 test [p icking thedaisies , when sudden ly a white rabbit with pink eyes ran]  
> 99 test [close by her.]  
> 
> ...
> 
> estimated hyperparameters of NPYLM  
> HPYLM theta [2.177930383536845e-05 1.4599305415060886e-05]  
> HPYLM d [0.5250034491443583 0.7908038176749297]  
> VPYLM theta [0.0009415115392966703 0.0004084690004264025 0.00024629651557873033 0.00029730160354459205 4.271234793924164 0.22679666227061274 0.8695756355064108 0.7033471705263717 0.3054603909967413 1.1323162367240984 0.3051399527625599 0.2630477155334784 0.05854985233273964]  
> VPYLM d [0.20158370699264577 0.624128723574685 0.7779914555133184 0.922945657702193 0.5908393230605639 0.5992113604167763 0.5815572048158039 0.6200554416938495 0.6231084007758823 0.11453709219091775 0.9256009234120085 0.013547750030616427 0.18586826282838004]  
> VPYLM alpha 4  
> VPYLM beta 1  

## References
- (Uchiumi et al., 2015) https://www.aclweb.org/anthology/P15-1171.pdf

## License