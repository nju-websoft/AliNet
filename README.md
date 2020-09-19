# [Knowledge Graph Alignment Network with Gated Multi-hop Neighborhood Aggregation](https://aaai.org/ojs/index.php/AAAI/article/view/5354)

> Graph neural networks (GNNs) have emerged as a powerful paradigm for embedding-based entity alignment due to their capability of identifying isomorphic subgraphs. However, in real knowledge graphs (KGs), the counterpart entities usually have non-isomorphic neighborhood structures, which easily causes GNNs to yield different representations for them. To tackle this problem, we propose a new KG alignment network, namely AliNet, aiming at mitigating the non-isomorphism of neighborhood structures in an end-to-end manner. As the direct neighbors of counterpart entities are usually dissimilar due to the schema heterogeneity, AliNet introduces distant neighbors to expand the overlap between their neighborhood structures. It employs an attention mechanism to highlight helpful distant neighbors and reduce noises. Then, it controls the aggregation of both direct and distant neighborhood information using a gating mechanism. We further propose a relation loss to refine entity representations. We perform thorough experiments with detailed ablation studies and analyses on five entity alignment datasets, demonstrating the effectiveness of AliNet.

<p align="center">
  <img width="90%" src="https://github.com/nju-websoft/AliNet/blob/master/architecture.png" />
</p>

## Dataset
We use two entity alignment datasets DBP15K and DWY100K in our experiments. DBP15K can be downloaded from [JAPE](https://github.com/nju-websoft/JAPE) and DWY100K is from [BootEA](https://github.com/nju-websoft/BootEA).


## Code

* "alinet.py" is the implementation of AliNet (with relation loss and iterative neighborhood augmentation).

### Dependencies
* Python 3
* Tensorflow 2.0 (**Important!!!**) 
* Scipy
* Numpy
* Pandas
* Scikit-learn

### Running

For example, to run AliNet on DBP15K ZH-EN, use the following script (supposed that the DBK15K dataset has been downloaded into the folder '../data/'):
```
python3 main.py --input ../data/DBP15K/zh_en/mtranse/0_3/
```

To run AliNet on DBP15K, use the following script:
```
bash run_dbp15k.sh
```

To run AliNet (w/o iterative neighborhood augmentation) on DBP15K ZH-EN, use the following script:
```
python3 main.py --input ../data/DBP15K/zh_en/mtranse/0_3/ --sim_th 0.0
```

To run AliNet (w/o relation loss and neighborhood augmentation) on DBP15K ZH-EN, use the following script:
```
python3 main.py --input ../data/DBP15K/zh_en/mtranse/0_3/ --rel_param 0.0 --sim_th 0.0
```

> If you have any difficulty or question in running code and reproducing experimental results, please email to zqsun.nju@gmail.com or cmwang.nju@gmail.com.

## Citation
If you use our model or code, please kindly cite it as follows:      
```
@inproceedings{AliNet,
  author    = {Zequn Sun,
               Chengming Wang, 
               Wei Hu, 
               Muhao Chen, 
               Jian Dai, 
               Wei Zhang, 
               Yuzhong Qu},
  title     = {Knowledge Graph Alignment Network with Gated Multi-Hop Neighborhood Aggregation},
  booktitle = {AAAI},
  pages     = {222--229},
  year      = {2020}
}
```
