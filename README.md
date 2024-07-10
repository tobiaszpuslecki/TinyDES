# TinyDES
## _Dynamic Ensemble Selection on Tiny Devices_

We examine a DES-Clustering approach for a multi-class computer vision task within TinyML systems. This method allows for adjusting classification accuracy, thereby affecting latency and energy consumption per inference. We implemented the TinyDES-Clustering library, optimized for embedded system limitations. Experiments have shown that a larger pool of classifiers for dynamic selection improves classification accuracy, and thus leads to an increase in average inference time on the TinyML device.

Using the DESlib framework, the DESClustering method is trained. Then, using an implemented converter, this classifier is ported to C language, as it is commonly used in embedded systems. 

TinyDES-Clustering library can potentially be used in any TinyML application where balancing accuracy and energy consumption is crucial.

## Installation

Install the dependencies:

```sh
pip install -r requirements.txt
```
