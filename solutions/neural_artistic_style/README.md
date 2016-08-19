# Neural Algorithm of Artistic Style

This example shows the implementation of of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576). They use deep learning to extract the style of a picture and transfer it to a new image. 


## Download data
To download the data:

	cd data
	python download_data.sh

## Run Neural Algorithm for Artistic Style

To run the algorithm in a GPU (see `--help` for more options):
    
	cd python/mxnet  
	python neural_style.py --gpu 0 

# Results
We can combine a picture of Bill Gates with the style of Vincent van Gogh's Starry Night:

<img src="data/bill-gates-desk.jpg" alt="Bill Gates" style="width: 50px;">
<img src="data/starry_night.jpg" alt="Starry Night" style="width: 50px;">
<img src="data/bill-gates-desk-starry.jpg" alt="Bill Gates with Starry Night style" style="width: 50px;">

