# EQB2A
Source code for the paper "Efficient Query-based Black-Box Attack against Cross-modal Hashing Retrieval".

## Requirements
* python == 3.7.10
* pytorch == 1.4.0
* torchvision == 0.2.1
* numpy == 1.19.2
* h5py == 3.4.0
* scipy == 1.7.1

## Datasets
We used three cross-modal datasets for experiments. Since MS-COCO do not have common text features, we use the pre-trained BERT model to extract 1024-dimension text features. All datasets are available by the following link:

* FLICKR-25K: https://pan.baidu.com/s/1Ie9PDqC9mAmBdxqX0KJ0ng <br> Password: yjkd
* MS-COCO: https://pan.baidu.com/s/1ocZTVx1GFFdceoSYbIWkbQ <br> Password: 2a6l
* NUS-WIDE: https://pan.baidu.com/s/1Yvqt4Bdjsq1gPaJn2IqIEw <br> Password: doi1

## Attacked models
We carry out targeted adversarial attack for six cross-modal hashing methods, including three supervised methods (DCMH, CPAH, DADH) and three unsupervised methods (DJSRH, JDSH, DGCPN). All attacked hashing models can be obtained by the following link:

* Deep Cross-Modal Hashing (DCMH): https://github.com/WendellGul/DCMH
* Consistency-Preserving Adversarial Hashing (CPAH): https://github.com/comrados/cpah
* Deep Adversarial Discrete Hashing (DADH): https://github.com/Zjut-MultimediaPlus/DADH
* Deep Joint-Semantics Reconstructing Hashing (DJSRH): https://github.com/zs-zhong/DJSRH
* Joint-modal Distribution-based Similarity Hashing (JDSH): https://github.com/KaiserLew/JDSH
* Deep Graph-neighbor Coherence Preserving Network (DGCPN): https://github.com/Atmegal/DGCPN

## Citation
Coming soon...
