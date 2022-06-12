# Variational Graph Autoencoders

JAX implementation of [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) (Kipf and Welling, 2016).

_Disclaimer: A good chunk of the code, especially those related to dataset preprocessing, is attributed to this amazing [Colab tutorial](https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb) by Lisa Wang and Nikola Jovanović._

### Dependencies

- [JAX](https://jax.readthedocs.io/en/latest/)
- [Jraph](https://github.com/deepmind/jraph)
- [Haiku](https://github.com/deepmind/dm-haiku)
- [Optax](https://github.com/deepmind/optax)
- NetworkX
- scikit-learn

### Usage

Run either GAE (Graph Autoencoder) or VGAE (Variational GAE):

```zsh
python3 train.py --is_vgae=True
```

### Todos

- [ ] Make a runnable notebook with loss plots, etc.
- [ ] Try on different datasets
- [ ] Compare ROC-AUC with results in Kipf-Welling paper
- [ ] Add documentation comments for encoder, decoder, etc.
