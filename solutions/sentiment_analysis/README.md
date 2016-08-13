# Sentiment Analysis

Here we show some sentiment analysis examples with Recurrent Neural Networks and Convolutional Neural Networks.

## Bidirectional LSTM for IMDB database:
This file trains a Bidirectional LSTM and train it with the IMDB database.
    
        cd python/keras  
        python bilstm_imdb.py  

## Sentiment analysis using CNN:
Implementation of the [paper](http://arxiv.org/abs/1606.01781) Very Deep Convolutional Networks for Natural Language Processing, Conneau et al., 2016. The authors present an architecture for text processing which operatesdirectly on the character level and uses only small convolutions and pooling operations. The authors claim that it is the ﬁrst time that very deep convolutional nets have been applied to NLP. They surpass the state of the art accuracy in several public databases. 

Initially some libraries must be installed:

		pip install tqdm

In order to download the database Amazon review polarity:
	
		cd data
		python download_amazon_review_polarity.py  

To run the code:

		cd python/mxnet
		python very_deep_cnn_for_nlp.py