# Interpreting Unsupervised Anomaly Detection in Security via Rule Extraction

This is the project site of our paper "Interpreting Unsupervised Anomaly Detection in Security via Rule Extraction" (to appear in NeurIPS '23). The code is primarily developed by Ruoyu Li and Yu Zhang.

![image-20231226170630681](C:\Users\EliYu\AppData\Roaming\Typora\typora-user-images\image-20231226170630681.png)

### Abstract

Many security applications require unsupervised anomaly detection, as malicious data are extremely rare and often only unlabeled normal data are available for training (i.e., zero-positive). However, security operators are concerned about the high stakes of trusting black-box models due to their lack of interpretability. In this paper, we propose a post-hoc method to globally explain a black-box unsupervised anomaly detection model via rule extraction. First, we propose the concept of distribution decomposition rules that decompose the complex distribution of normal data into multiple compositional distributions. To find such rules, we design an unsupervised Interior Clustering Tree that incorporates the model prediction into the splitting criteria. Then, we propose the Compositional Boundary Exploration (CBE) algorithm to obtain the boundary inference rules that estimate the decision boundary of the original model on each compositional distribution. By merging these two types of rules into a rule set, we can present the inferential process of the unsupervised black-box model in a human-understandable way, and build a surrogate rule-based model for online deployment at the same time. 
We conduct comprehensive experiments on the explanation of four distinct unsupervised anomaly detection models on various real-world datasets. The evaluation shows that our method outperforms existing methods in terms of diverse metrics including fidelity, correctness and robustness. NIPS paper is at \url{[Interpreting Unsupervised Anomaly Detection in Security via Rule Extraction | OpenReview](https://openreview.net/forum?id=zfCNwRQ569)}.





```bash
@inproceedings{li2023interpreting,
  title={Interpreting Unsupervised Anomaly Detection in Security via Rule Extraction},
  author={Li, Ruoyu and Li, Qing and Zhang, Yu and Zhao, Dan and Jiang, Yong and Yang, Yong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```



## Installation

```bash
git clone https://github.com/Ruoyu-Li/UAD-Rule-Extraction
```

