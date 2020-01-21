# Seq2Seq Learning with Encoder-Decoder Neural Network


## Introduction:

One of the most important topics faced by decision makers in corporate and government agencies is their units’ future performance. This topic is becoming more and more addressable yet challenging with the advent of the big data era. In the past, using regression or ARIMA model might be enough to obtain a predictive result that is good enough for simple forecasting problems. However, as time series data gets more erratic and complicated, deep learning methods is playing a more and more important role in time series forecasting, since they make no assumption on the underlying patterns and are more robust to deal with noise in data. **Sequence to Sequence Learning with Encoder-Decoder Neural Network techniques** is actually a perfect fit for solving time series forecasting problems. 

This is a topic I tried to learn via a [Kaggle competition](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) when I was completing a project during my study in M.S. Business Analytics program at University of Minnesota. The key question for this competition is how to accurately predict the number of customers who visit the restaurant each day. But there are limited applications available online specifically about this topic. This specific modeling solution was referenced from this [Kaggle submission](https://www.kaggle.com/ievgenvp/lstm-encoder-decoder-via-keras-lb-0-5/output#L505) with some modifications. Many thanks to this well-organized script, I was able to recreate this encoder-decoder model. I would walk through the logic behind that solution here and hopefully it can benefit some folks who just started to step into this amazing field.


## Main files:

`How it works.md` - Main file that explains how this model works

`Seq2Seq (LSTM).ipynb`- Main notebook script that implement the modeling process

## How to reproduce competition results:
1. Download input files from the `data` folder - `train_final.zip`, `test_final.csv`, and `sample_submission`.
2. Download the notebook file called `Seq2Seq (LSTM).ipynb` and put it into the same local folder as the data files.
3. Run the whole notebook file.

## References:
>Seq2Seq

https://www.kaggle.com/ievgenvp/lstm-encoder-decoder-via-keras-lb-0-5

https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

>LSTM

https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577

>Attention

https://distill.pub/2016/augmented-rnns/

http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/


## Contact:

If there is anything to be corrected or you have any thoughts to share with me on this topic, please feel free to reach out! It's allways pleasure to learn more.
>Email: olivia.liang032@gmail.com

