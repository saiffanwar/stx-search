##### This repository is an adaptation of the open source LibCity Library, to which features have been added to allow for model explainability and transparency. The original repository docs can be found at the links below.



![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/logo.png)

------

[![ACM SIGSpatial](https://img.shields.io/badge/ACM%20SIGSPATIAL'21-LibCity-orange)](https://dl.acm.org/doi/10.1145/3474717.3483923) [![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/) [![Pytorch](https://img.shields.io/badge/Pytorch-1.7.1%2B-blue)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE.txt) [![star](https://img.shields.io/github/stars/LibCity/Bigscity-LibCity?style=social)](https://github.com/LibCity/Bigscity-LibCity/stargazers) [![fork](https://img.shields.io/github/forks/LibCity/Bigscity-LibCity?style=social)](https://github.com/LibCity/Bigscity-LibCity/network/members) 

# LibCity

[HomePage](https://libcity.ai/) | [Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/index.html) | [Datasets](https://github.com/LibCity/Bigscity-LibCity-Datasets) | [Conference Paper](https://dl.acm.org/doi/10.1145/3474717.3483923) | [Full Paper](https://arxiv.org/abs/2304.14343) | [Paper List](https://github.com/LibCity/Bigscity-LibCity-Paper) | [Experiment Tool](https://github.com/LibCity/Bigscity-LibCity-WebTool) | [EA&B Track Paper](https://arxiv.org/abs/2308.12899) | [中文版](https://github.com/LibCity/Bigscity-LibCity/blob/master/readme_zh.md) 



## The additional contribution to this repository is in the form of XAI method STX-Search, outlined in the paper below.

'*STX-Search: Explanation Search for Continuous Dynamic Spatio-Temporal Models*' https://www.arxiv.org/pdf/2503.04509



### Training a model

The method is implemented for the Traffic State prediction functions of the library to explain predictions made using Spatio-Temporal models predicting traffic speed, volume, etc. An example dataset is provided as the 'GRID' dataset which is a 10x10 grid road network with traffic simulated using the [SUMO](https://sumo.dlr.de/docs/index.html) simulation tool. 

For example, a TGCN model can be trained using the following command. Further details regarding model training can be found in LibCity documentation.

`python run_model.py --dataset GRID --model TGCN`

A trained model is saved at the following path:

`libcity/cache/1/model_cache/{MODEL}_{DATASET}.m`



### Explaining a model

The explainer is run using the following command to explain the prediction for node 5 in the graph using the 50 most important events:

`python libcity_explainer_runner.py --dataset GRID --model TGCN --target_node 5 --subgraph_size 50`



#### Visualisation:

To view the progress and the outcomes of the explanation framework, the following dash tool can be used

`python app.py --dataset GRID --model TGCN`