The present lsd (Learning on Synthetic Data) package provides an easy framework to run learning experiments on synthetic datasets. In particular, it implements USV-layers to allow for entropy computation during learning, using the heuristics replica formula from statistical physics as proposed in [1]. The computation of entropies requires the installation of the dedicated package [dnner](https://github.com/sphinxteam/dnner) (DNNs Entropy from Replicas).

**Install package**

To install the package simply run.
```
python setup.py install --user
```

**Run the code**

A few examples are provided in the `examples` folder.
Note that the path to a dedicated `data/` folder should be specified to the "prefix" kwarg.

**Reference**

[1] M. Gabrié, A. Manoel, C. Luneau, J. Barbier, N. Macris, F. Krzakala & L. Zdeborová, Entropy and mutual information in models of deep neural networks, [arXiv:1805.09785](https://arxiv.org/abs/1805.09785).
