# Highlights

* SAJ is a novel deep learning model for imputation of missing values in water quality data.
* SAJ integrates innovative approaches: RMSNorm, Lower Triangular Masking Multi-Head Attention, and Dimension Splitting, which have not been explored in any imputation tasks before.
* The model is tested on real world water quality data of different sizes and high missing rates.
* SAJ exhibits state-of-the-art achievements on imputation tasks of water quality data.

# README
It is a pytorch implementation of the paper "SAJ: A Deep Learning Approach to Missing Data Imputation in Water Quality Datasets, Ishan Prasad Banjara, Deepesh Upreti, Kalam Pariyar, Suman Poudel, Shukra Paudel".

In the present age of environmental degradation, extracting meaningful insights from large amount of water quality data is crucial for minimizing the effect of anthropogenic activities on dwindling water resources. A key problem encountered is the lack of accurate and reliable data, especially caused by high missingness, which impairs the ability of decision makers to take timely actions for mitigating the environmental damage. The study aims to introduce a deep learning (DL) model named as SAJ: Self-Attention Joint with convolution, for efficient imputation of missing water quality data, under very high missingness scenarios (~ 90%). SAJ is an innovative DL model encapsulating convolutional neural network and self-attention mechanism along with integration of lower triangular masking and RMS normalization technique, all of which provides the architectural novelty to the model. For two different water quality datasets, the model outcompetes other State-of-the-Art (SOTA) models, exhibiting lower value of error metrics which substantiates its excellence in imputation tasks. Similarly, SAJ demonstrated reduced average inference time and number of parameters, further validating its superiority. Finally, the model also holds the promise for further improvements, indicating its potential for dominating the domain of water quality data imputation by even greater margin.

