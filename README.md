# My bachelor's thesis:

### Modeling of visual recognition based on spiking neural networks with a competition between local receptive fields

# Overview

I work with unsupervised learning on MNIST of Spiking Neural Networks.

I've reproduced the results of this [paper](https://arxiv.org/abs/1904.06269) using the [bindsnet](https://github.com/Hananel-Hazan/bindsnet) library. It is important to read the paper for further understanding.

I have two goals:

* Comparison of Locally Connected networks to Convolution and Fully Connected networks (with similar  number of parameters)
* Finding an efficient way of training inhibitor (competitive) connections and finding out if they positively affect the accuracy.

Currently I have trained 25- and 100-filter Locally Connected networks. The networks have two layers: X - input, Y - output (hidden layer). The Y neurons with the same receptive fields have competitive (inhibitor) connections between them. These connections are defined as a constant negative weight, but can be trained later. Training of these competitive connections to improve accuracy is the main goal of this work, but I also compare Locally Connected networks with Convolution networks and Fully Connected networks. Lower are the results of this comparison. Best results are presented with std ~ 0.1-1%. Some of the lower accuracies might be for networks with sub-optimal parameters, because I didn't have enough time for all the computations.

| type    | n\_filters | kernel\_size | accuracy |
| ------- | ---------- | ------------ | -------- |
| LC\_SNN | 100        | 12           | 0\.896   |
| LC\_SNN | 100        | 8            | 0\.814   |
| LC\_SNN | 25         | 12           | 0\.797   |
| LC\_SNN | 25         | 8            | 0\.75    |
| C\_SNN  | 100        | 8            | 0\.79    |
| C\_SNN  | 25         | 12           | 0\.668   |
| C\_SNN  | 25         | 8            | 0\.665   |
| FC\_SNN | 100        | 20           | 0.734    |

I'm currently experimenting with training of YY inhibitory connection. I have some positive results, they will be here soon.

I found an interesting way to get good accuracy after training YY. In my method I first lock the YY connection (to some value like -100) and only train XY. Then I clear YY weights, make them zeros, lock XY weights and train YY (the lowest are clamped to some value). The resulting weights look something like this:

![competitive weights](overview/weights_YY_cl.png)

Their distribution:

![competition distribution](overview/competition_distribution.png)

Locally Connected don't need a lot of training examples:

![accuracy dependence on number of training iterations](overview/accuracy-n_iter.png)

Here are 100-filter Locally Connected weights after 5000 iterations of training:

![Weights XY](overview/weights_XY.png)

The overall accuracy of this network is 0.89. Here is the accuracy distribution between different labels:

![Accuracy distribution](overview/accuracy_distribution.png)

And here is the confusion matrix:

![Confusion matrix](overview/confusion_matrix.png)

Statistically calculated votes based on mean spiking activity of neurons for each label give us the following  distribution: 

![Votes distribution](overview/votes_distribution.png)

On the figure above 1, ..., 10 means "best class for the neuron", "second best class for the neuron", .. , "worst class for the neuron".

In the paper I'm reproducing only the top3 classes are used in the voting mechanism. I am using all top10 classes, which a bit improves the accuracy and doesn't take much more time.

![accuracy-top_n](overview/accuracy-top_n.png)

Here are the results of a gridsearch for Locally Connected networks with 100 filters performed over mean weight per Y neuron and competitive weight parameters:

![gridsearch results](overview/gridsearch-LC_SNN.png)

and [interactive 3D plot](overview/gridsearch-LC_SNN.html) (download the file and open in your browser)

The results for 25 filters (Locally Connected network):

![gridsearch results 25](overview/gridsearch-LC_SNN-25.png)

and [interactive 3D plot](overview/gridsearch-LC_SNN-25.html) 

And the results of gridsearch for 25 filters Convolution Networks:

![gridsearch results](overview/gridsearch-C_SNN.png)

and [interactive 3D plot](overview/gridsearch-C_SNN.html)

As the figures above show, Locally Connected networks can achieve around 14% better accuracy than Convolution Networks with the same number of filters. 

# Work to do

* Compare to a network with trainable competition weights. Right now I'm searching for good training parameters.

# Usage

To reproduce my results download this repository and install all required packages.

Install PyTorch with Anaconda with the command from the [official website](https://pytorch.org/).

Then run

```
pip install -r requirements.txt
```

All my code is located in the **thesis** directory.

The **bindsnet** directory contains modified and corrected [bindsnet](https://github.com/Hananel-Hazan/bindsnet) package.

Pre-trained networks are located in the **networks** directory. Choose branch **no-networks** if you don't want them (~800 MB folder).

Run the following code in Jupyter Notebook. The notebook must be located at the root of the project.

## Basic imports

```python
from thesis.nets import LC_SNN, C_SNN
from thesis.utils import view_database, load_network, delete_network
```

To view available networks use

```python
view_database()
```

Output:

| name                                                     | accuracy | n_iter | mean_weight         | n_filters | c_w    | crop | kernel_size | stride | time_max | intensity |
| -------------------------------------------------------- | -------- | ------ | ------------------- | --------- | ------ | ---- | ----------- | ------ | -------- | --------- |
| 077029b0df623416d0640d0d400fada60a5997c9f1864dfe0ffc0848 | 0.8616   | 5000   | 0.24                | 100       | -50.0  | 20   | 12          | 4      | 250      | 127.5     |
| 10706382198294901892                                     | 0.7655   | 10000  | 0.49000000000000005 | 25        | -100.0 | 20   | 12          | 4      | 250      | 127.5     |
| 11078776799026513062                                     | 0.7694   | 10000  | 0.5                 | 25        | -100.0 | 20   | 12          | 4      | 250      | 127.5     |
| 11579302362096645865                                     | 0.7679   | 10000  | 0.40000000000000013 | 25        | -100.0 | 20   | 12          | 4      | 250      | 127.5     |
| 12b1568c093bf5563e169f4e864154d20b95cec6492c59e0e2295068 | 0.7443   | 10000  | 0.48                | 100       | -20.0  | 20   | 12          | 4      | 250      | 127.5     |
| 146895984239560197                                       | 0.7533   | 10000  | 0.5000000000000001  | 25        | -100.0 | 20   | 12          | 4      | 250      | 127.5     |

## Loading an existing network

Copy the name of a network you want to load.

```python
net = load_network('3a2846ea8cac0ceae1e970206cb6f0de92e5e6c2b930e99beff81ab4')
```

Network loaded. Now you can check it's behavior:

```python
net.feed_label(5, plot=True)
```

Output:

Prediction: 5

![Input image 3](overview/input_image_3.png)

![Spikes](overview/spikes_Y.png)



![Best voters](overview/best_voters.png)

![best voters voltage](overview/best_voters_voltage.png)

![random neuron voltage](overview/random_neuron_voltage.png)

## Training a new network

Run with desired parameters:

```python
net = LC_SNN()  # C_SNN() to create a convolution network
```

c_l = True will make the competition weights trainable.

Then to train the network (and be able so see the progress) run

```python
net.train(n_iter=5000, plot=True, vis_interval=30)  # max is 50000, 5000 is fine 

net.calibrate(n_iter=5000)  # max is 10000, 5000 is fine

net.calculate_accuracy(n_iter=1000)  # max is 10000
```

To calibrate and calculate accuracy with a linear classifier add use .calibrate_lc() and .calculate_accuracy_lc()

The network is ready. To save the network:

```python
net.save()
```

To check network's accuracy, accuracy distribution, confusion matrix, and votes distribution you can use:

```python
accuracy = net.accuracy

accs, fig_accs = net.accuracy_distribution()
fig_accs.show()

fig_conf = net.confusion()
fig_conf.show()

fig_votes_distr = net.votes_distribution()
fig_votes_distr.show()
```

It is also possible to run the network over custom input images:

```python
net.feed_image('image.png')
```



## Deleting a network

```python
delete_network(net.name)
```



## Dash Application

I've also made a simple Plotly Dash application which can be used to observe the training process of a network. I'm not updating it very often, so it might be broken at the moment, if so I'll fix it after all the important work is done.