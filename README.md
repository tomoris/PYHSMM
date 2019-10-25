# PYHSMM with golang
Implementation of Pitman-Yor Hidden Semi-Markov Model (PYSHMM) (Uchiumi et al., 2015) with golang. PYHSMM is a non-parametric Bayesian model for inducing words and part-of-speech tags from unsegmented texts without annotated data.

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
go get github.com/tomoris/PYHSMM.git
cd $GOPATH/src/github.com/tomoris/PYHSMM
go build main.go
```

### for Python Extention (if you want)
```
gopy build -vm=`which python3` -output=pylib github.com/tomoris/PYHSMM/bayselm
```

## Usage
Training language model.  
`./main lm --model hpylm --maxNgram 2 --trainFile data/alice.train.txt --testFile alice.test.txt`  
Training word segmentation model without labeled data.  
`./main ws --model npylm --maxNgram 2 --trainFile data/alice.raw --threads 8 -- saveFile alice_npylm.model`  

### for Python Extention (if you want)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./pylib
export PYTHONPATH=$PYTHONPATH:./pylib
go build -o trainWordSegmentation.so -buildmode=c-shared main.go
python train_ws.py --model npylm --maxNgram 2 --train_file data/alice.raw
```



### Models
```
ngram: Interporated n-gram model
hpylm: Hierarchical Pitman-Yor Language Model
vpylm: Variable order HPYLM
npylm: Nested Pitman-Yor Language Model (unsupervised word segmentation model)
pyhsmm: Pitman-Yor Hidden SemiMarkov Model (unsupervised word segmentation and POS induction model)
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
| pyhsmm |    7294.0 |    2 |                 10 |

### Unsupervised Word Segmentation

| Model  | Precision | Recall | F-score |
| :----: | --------: | -----: | ------: |
| npylm  |         ? |      ? |       ? |
| pyhsmm |         ? |      ? |       ? |


## References
- (Uchiumi et al., 2015) https://www.aclweb.org/anthology/P15-1171.pdf

## License