# Seq2Seq Learning with Encoder-Decoder Neural Network


## Introduction:

This is a project completed during my study in M.S. Business Analytics program at University of Minnesota - Fall 2019 course in Predictive Analytics (supervised learning). We managed to implement a solution for a [time series forcasting competition problem on Kaggle](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) using stacked generalization with XGBoost and LSTM (Seq2Seq with Encoder-Decoder) as 2 based learners, which result in final performance of .5000 RMSLE. My colleagues Sam Musch has created a well-organized project summary in his [github link](https://github.umn.edu/MUSCH038/Predictive-Project---Time-Series), including data cleaning, feature engineering, and model training. Therefore, I would like to use this github to give a practical introduction/explanation specifically to the Sequence to Sequence Learning with Encoder-Decoder Neural Network techniques on time series forcasting problem, using the data in this project as an example. 

This specific modeling solution was referenced from one of [the Kaggle submissions](https://www.kaggle.com/ievgenvp/lstm-encoder-decoder-via-keras-lb-0-5/output#L505) with some modifications. Many thanks to this well-organized script, I was able to recreate an encoder-decoder model. However, there was not much detailed/intuitive explanation about the motivation and the structure of the model, which made it hard for me as a beginner in this field to follow at the first place. Therefore, I would like to walk through the logic behind the solution here and hopefully it can benefit some folks who just started to step into this amazing field.


## Main Files:

`How it works.md` - Main file that explains how this model works

`Seq2Seq (LSTM).ipynb`- Main notebook script that implement the modeling process

## How to reproduce competition results:
1. Download input files from the `data` folder - `train_final.zip`(needs to be decompressed), `test_final.csv`, and `sample_submission`
2. Download the notebook file called `Seq2Seq (LSTM).ipynb` and put it into the same local folder as the data files.
3. Run the whole notebook file

## References:
>Seq2Seq

https://www.kaggle.com/ievgenvp/lstm-encoder-decoder-via-keras-lb-0-5

https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

>LSTM

https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577

>Attention

https://distill.pub/2016/augmented-rnns/

http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/

