# Word Prediction from Characters

This example shows how to use Recurrent Neural Networks to learn the structure of words and generate them using characters as inputs. It uses a dataset of words in English to learn the language. 

## Download the Dataset 

To download the data:

       cd data
       python download_data.py

## Run Recurrent Neural Networks

This example shows how to use a RNN, specifically a LSTM, to learn words from characters. 

* Word Prediction with LSTM:
    
        cd R/mxnet  
        R < word_prediction_lstm.R --no-save  

