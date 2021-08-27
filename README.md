# Alleviating Exposure Bias via Contrastive Learning for Abstractive Text Summarization

This repository is the implementation for the paper [Alleviating Exposure Bias via Contrastive Learning for Abstractive Text Summarization](https://arxiv.org/abs/2108.11846).

Some of my codes come from the Transformers example in https://github.com/huggingface/transformers/tree/v4.0.1/examples/seq2seq.

## Requirements
- python 3.7
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/) 4.1.1 

### Install requirements

```
pip install -r requirements.txt
```

## Train and Test

### Xsum

- Get dataset

```
wget https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz
tar -xzvf xsum.tar.gz
```
- Run 
```
bash exp/run_consum_xsum.sh
```

### CNNDM:

- Get dataset

```
wget https://cdn-datasets.huggingface.co/summarization/pegasus_data/cnn_dailymail.tar.gz
tar -xzvf cnn_dailymail.tar.gz
mv cnn_dailymail/validation.source cnn_dailymail/val.source 
mv cnn_dailymail/validation.target cnn_dailymail/val.target 
```
- Run 
```
bash exp/run_consum_cnndm.sh
```

### Multi-News:

- Get dataset

```
wget https://cdn-datasets.huggingface.co/summarization/pegasus_data/multi_news.tar.gz
tar -xzvf multi_news.tar.gz
mv multi_news/validation.source multi_news/val.source 
mv multi_news/validation.target multi_news/val.target 
```
- Run 
```
bash exp/run_consum_multinews.sh
```
