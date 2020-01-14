# How it works



### What is Seq2Seq?

First, for those who are not familiar with Seq2Seq concept, I would like to give a brief introduction about it. Borrowing the definition from [a Keras Blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) : Seq2Seq, short for Sequence-to-sequence learning, is about training models to convert sequences from one domain (e.g. sentences in English) to sequences in another domain (e.g. the same sentences translated to French). It has been widely used in the following subjects:

- Neural Machine Translation
- Speech Recognition
- Text Summarization
- Image Captioning



Simply using RNN for this many-to-many forecasting problem does not work as well as adding seq2seq to the RNN model for most cases mentioned above. Since a simple RNN assumes that it is possible to generate `target[...t]` given `input[...t]`. However, as pointed in that blog, information about the entire input sequence is necessary in order to start generating the target sequence in the general case. Therefore, a more advanced setup (seq2seq) is created to solve this problem. Detailed explanation on how it works on language translation problem is described in the blog mentioned above, thus I will only give a brief idea about it here. But detailed description about its applications on time series data will be given later on this page.



In order to capture the whole information or changing pattern of the data in previous steps, we will use a RNN layer (or stack thereof) as **"encoder"** to process the raw input data into some representations of it, which "encodes" the information in the original input data. Then we use another RNN layer (or stack thereof) as **"decoder"** to produce the expected output conditioned on the representations generated from the encoder layer.



### Why does it fit time series problem?

It might be more intuitive to understand why this concept is needed in a language translation problem, since most translation is not working in the way of "word to word". However, seq2seq is also very helpful in solving time series forecasting problems, even though there is only a few applications available online on this as I am aware of. This implementation is inspired by the first place solution for another time series prediction on Kaggle. I would recommend readers to go through [that winner's post](https://github.com/Arturus/kaggle-web-traffic) for a deeper understanding on this type of topic and some more advanced settings after reading my page. 



In the general case, time series prediction mainly depends on the learning of seasonality in data. Forecasting for a specific date will depends on not only learning one day's data but some weekly/yearly trends. Therefore, seq2seq seems natural for this task, since we will predict the next value conditioned on the joint probability of the previous ones as well as our past predictions. In this case, the probability of getting affected by some extreme data would be likely to be minimized. **"Teacher forcing"** method was used in my case to improve model skill and stability, since it will use the ground truth from a prior time step as input instead of using the best guess that has the maximum probability.



RNN is built for sequential problems like this to provide the network a memory of the information recognized for better prediction in the next step. Besides its natural advantage for sequential prediction, RNN, as neural network, is powerful enough to discover and learn features on its own, even if we were using some basic features. And any exogenous features can be This is the best model I would want to use when I am not confident enough with my existing features. 



### Feature engineering



Considering the input data structure the LSTM RNN model needed, we filled up all the dates within the whole time span (2016-01-01 ~ 2017-05-31) for each stores with number of visitors as 0 on those dates, and the time-independent features (food types, longitude, latitude, etc) are "stretched" to timeseries length. 

**Labels:**

`air_store_id` - The unique store id in the Air system provided in the original data

`Visit date` - original column in the data. The date visitor come to the restaurant. Used for mapping different data files

**Target attribute:**

`# of visitors` - The number of visitors for the restaurant on a specific date



The following are the features included in the model:

**Time dependent features -**

`Day of week` - day of the week. Use numeric value to represent. If the day is weekend, it might have more traffic than normal weekdays.

`Month & week of the year` - different months/week also has different volume. This variable showcases the seasonality of the time in a year

`is_holiday, next_day, prev_day` - whether a day/ the next day/ the date in the previous year is holiday. This variable can indicate the holiday flag. If it's holiday time, there might be higher traffic outside and restaurant might also have more number of visitors. 

`Days since 25th` - The next feature calculates how many days it has been since the previous 25th of the month. The 25th is special because this is when most Japanese people receive their monthly paycheck ([Japan Visa]([http://www.japanvisa.com/news/japan-payroll-%E2%80%93-introduction](http://www.japanvisa.com/news/japan-payroll-–-introduction))).

`Consecutive Holidays` - we believe consecutive holidays and the length of days off-work also have a say in restaurant visiting patterns. For example, if the holiday is Friday, we will mark Friday, and the followed weekend with 3.

`# of visitors in t-1,...,t-7` - The number of visitors on `t-2,...,t-7` day 



**Store dependent features -**

`food_type` - The type of major food provided by the restaurant

`latitude, longitude` - location of the restaurant, which we believe indicates similar population and citizen visiting patterns as where restaurants locate.



**Time & store dependent features -** 

`prev_visitors` - The number of visitors in the same day of week, which in the same month of the year, in the previous year.

`mean, median, max, min visitors` - average/median/maximum/minimum number of visitors for each store for a specific weekday.

count_observations



As a side note here, I also encoded all the categorical features to and normalized all the numerical values to avoid the negative impact from different scales between features.



### Model core



##### Input data structure:

We are predicting the number of visitors for 829 restaurants on each day from 2017-04-23 to 2017-05-31 (39 days total) using the number of visitors and features of those restaurants on each day from 2016-01-01 to 2017-04-22(478 days total). 

We will need to reshape our training data into the input shape RNN requires, which is (samples, timesteps, features) corresponding to (batch_size, time_steps, input_dim).

Here I would use a graph to depict the structure better:

![input_shape](https://github.com/Olliang/Time-Series-Forcasting/blob/master/images/input_shape%20graph.png)

##### Modeling:

> > Main algorithm 



I assume you know how RNN and LSTM looks like, but let’s have a short reminder. LSTM was proposed mainly to address gradient vanishing problem typical naïve RNN suffers from, since the hidden states and cell states updated in each LSTM cell through 3 gates will help capture those relatively longer previous memory. 

![LSTM](https://github.com/Olliang/Time-Series-Forcasting/blob/master/images/LSTM network.png)

For deeper understanding of this concept, I recommend reading [Colah’s blog](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjGu_Kb9qHfAhUD0RoKHYSjAgwQFjAAegQIAhAC&url=http%3A%2F%2Fcolah.github.io%2F&usg=AOvVaw1GKlzuXDPgSuty6MfhWlol) for an in-depth review of LSTM networks. To summarize how LSTM works, I would specify some key concepts here to help refresh your memory for understanding the concepts used in the following model.

There are 3 different gats that regulate information flow in a LSTM cell - forget gate, input gate, and output gate. Each gate uses `sigmoid` activation function to decide what information should be kept or thrown away. `tanh` activation function is used to regulate the information passed through. Here I would borrow a graph that depicts clearly how hidden and cell states are processed and updated within a LSTM cell from [Nir Arbel's blog](https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577). And I also provided the simplified equations for calculating hidden and cell states below the graph.

![LSTM cell](https://github.com/Olliang/Time-Series-Forcasting/blob/master/images/LSTM%20cell.png)



> > Training-inference process

The model has two main parts: encoder and decoder. The encoder takes input features of 39 days (t*1, t*2 … t39) and encode their hidden states through LSTM neural network into a fixed length vector. Then it passes the hidden states to decoder. Decoder, designed as 2 fully connected LSTM neural networks, uses them with the features of 39 days shifted 1 day forward (t*2, t*3 … t40) to predict number of visitors per each of 829 restaurants in t_40. The LSTM layer is defined to return both sequences and state. The final hidden and cell states are ignored and only the output sequence of hidden states is referenced. 



To decode our encoded sequence, we will repeatedly:

- 1) Encode the input sequence and retrieve the initial decoder state
- 2) Run one step of the decoder with this initial state and a "start of sequence" value as target. The output will be the next target value.
- 3) Append the target value predicted and repeat.



The following graph depicts a **greedy encoding** model, which use the predicted value that has the maximum probability as a correct output of the current time step to be an input for the next time step.

![greedy decoding](https://github.com/Olliang/Time-Series-Forcasting/blob/master/images/model%20(Encoder%2BDecoder)-Greedy%20Decoding.png)

However, in order to improve model skill and stability as mentioned earlier, I used teacher forcing method for better model performance, which use the ground truth from a prior time step as input for the next time step.

The following graph depicts the improved model with **teacher forcing** method:

![teacher forcing](https://github.com/Olliang/Time-Series-Forcasting/blob/master/images/model%20(Encoder%20%2B%20Decoder)%20-%20teacher%20forcing.png)





### Further improvement



**Problems with encoder/decoder model:** Each decoder time step depends on the same fixed-size encoder representation, which means encoder representation is assumed to hold the information for every encoder time step.



**How to address:** Attention will help to address this problem and perform a better result. Attention works by removing the portion where we compute the encoder representation and doing it slightly differently. We calculate how close each decoder hidden state(s1,s2,...) to each encoder hidden state(h1, h2,...) with function s (e.g. dot product). And then put all those similarities into the `softmax` function to make them into probabilities. In this way, you can compare each state and find the most important states to give them more attention. Therefore, rather than simply taking the some the encoded representation of the whole time period as an input to the decoder, we will take each hidden state and multiply that by my guess as to how close each of them to the input of decoder(s1, s2,...). To summarize, Attention basically is a mechanism that lets the neural network focus its attention on some parts of the input to generate dynamic encoded representation for each decoding step. 



The following graph depicts an example of the process of Seq2Seq + attention model when producing the second time step in the sequence.

![attention](https://github.com/Olliang/Time-Series-Forcasting/blob/master/images/seq2seq%20%2B%20attention.png)



















