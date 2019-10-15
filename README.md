# PYHSMM with golang
Implementation of Pitman-Yor Hidden Semi-Markov Model (PYSHMM) (Uchiumi et al., 2015) with golang. PYHSMM is a non-parametric Bayesian model for inducing words and part-of-speech tags from unsegmented texts without annotated data.

## Prerequisites
```
Go: 12.5
gonum.org/v1/gonum/stat/distuv
github.com/cheggaaa/pb/v3
```
## Installing
```
go get github.com/tomoris/PYHSMM.git
cd $GOPATH/src/github.com/tomoris/PYHSMM
go build main.go
```

## Usage
Training language model.  
`./main lm --model hpylm --maxNgram 2 --trainFile alice.train.txt --testFile alice.test.txt`  
Training word segmentation without labeled data.  
`./main ws --model npylm --maxNgram 2 --trainFile alice.raw`  


### model
```
ngram: Interporated n-gram
hpylm: A Hierarchical Bayesian Language Model based on Pitman-Yor Processes
vpylm: variable order HPYLM
npylm: Nested Pitman-Yor Language Model (unsupervised word segmentation model)
pyhsmm: Pitman-Yor Hidden SemiMarkov Model (unsupervised word segmentation and pos induction model
```

## References
- (Uchiumi et al., 2015) https://www.aclweb.org/anthology/P15-1171.pdf

## License