# [Dravidian-Offensive-Language-Identification](https://arxiv.org/abs/2102.07150)

This repository contains the code and resources to reproduce the work of Team indicnlp@kgp's submission to the shared task ```Offensive Language Identification in Dravidian Languages``` at Dravidian Language Technology Workshop at EACL 2021

## Tokenizers and Language Models

The ULMFiT Language Models were inspired by the paper ```Gauravarora@HASOC-Dravidian-CodeMixFIRE2020: Pre-training ULMFiT on Synthetically
Generated Code-Mixed Data for Hate Speech
Detection``` ~ [Gaurav Arora](https://arxiv.org/pdf/2010.02094.pdf) and we borrow the pre-trained language models from their work. The links to the three code-mixed languages used in our work are:

1. [Tamil](https://github.com/goru001/nlp-for-tanglish)
2. [Malayalam](https://github.com/goru001/nlp-for-manglish)
3. [Kannada](https://github.com/goru001/nlp-for-kannada)

The language models and tokenizers need to be downloaded for using the ULMFiT notebooks.

## OLID Dataset

Apart from the dataset supplied by the organizers of the shared task, we also used a monolingual English Offensive Language Identification Dataset ([OLID](https://arxiv.org/pdf/1902.09666.pdf)) used in the SemEval-2019 Task 6 (OffensEval). The dataset contains the same labels as our task datasets with the exception of the ```not in intended language``` label. The one-to-one mapping between the labels in OLID and it's large size of 14k tweets makes it suitable for aiding the transfer learning.

## Transformer Architecture

The Transformer Architecture used by us is shown in the figure. We used the pre-trained models realeased by [HuggingFace](https://huggingface.co/transformers/pretrained_models.html).

![Transformer Architecture](https://github.com/kushal2000/Dravidian-Offensive-Language-Identification/blob/master/Transformer_Architecture.jpg)

## Results

Our final submission was an ensemble of mBERT, XLM-R and ULMFiT which ranked 1st, 2nd and 3rd on the Malayalam, Tamil & Kannada Datasets respectively. The weighted-F1 score was the metric of choice for this task. The breakdown for all models is also reported in our paper.

### ML Models

| Model         | Tamil | Malayalam | Kannada |
|---------------|-------|-----------|---------|
| Random Forest | 0.69  | 0.94      | 0.62    |
| Naive Bayes   | 0.74  | 0.94      | 0.64    |
| Linear SVM    | 0.74  | 0.95      | 0.65    |

### RNN Models

| Model        | Tamil | Malayalam | Kannada |
|--------------|-------|-----------|---------|
| Vanilla LSTM | 0.74  | 0.95      | 0.64    |
| ULMFiT       | 0.76  | 0.96      | 0.71    |

### Transformer Models

| Model      | Tamil | Malayalam | Kannada |
|------------|-------|-----------|---------|
| mBERT      | 0.74  | 0.95      | 0.66    |
| XLM-R      | 0.76  | 0.96      | 0.67    |
| mBERT (TL) | 0.75  | 0.97      | 0.71    |
| XLM-R (TL) | 0.78  | 0.97      | 0.72    |

*TL -- Transfer Learning used (details in our system description paper)


