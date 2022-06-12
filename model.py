import jax
import jax.numpy as jnp
import haiku as hk
import jraph

from typing import Tuple

def vgae_encoder(graph: jraph.GraphsTuple,
                 hidden_dim: int,
                 latent_dim: int) -> Tuple[jraph.GraphsTuple, jraph.GraphsTuple]:
  """VGAE network definition."""
  graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
  
  @jraph.concatenated_args
  def hidden_node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for hidden layer."""
    net = hk.Sequential([hk.Linear(hidden_dim), jax.nn.relu])
    return net(feats)

  @jraph.concatenated_args
  def latent_node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for latent layer."""
    return hk.Linear(latent_dim)(feats)

  net_hidden = jraph.GraphConvolution(
    update_node_fn=hidden_node_update_fn,
    add_self_edges=True
  )
  h = net_hidden(graph)
  
  net_mean = jraph.GraphConvolution(
    update_node_fn=latent_node_update_fn,
    add_self_edges=True
  )
  net_log_std = jraph.GraphConvolution(
    update_node_fn=latent_node_update_fn,
    add_self_edges=True
  )
  mean, log_std = net_mean(h), net_log_std(h)
  return mean, log_std
  

def gae_encoder(graph: jraph.GraphsTuple,
                hidden_dim: int,
                latent_dim: int) -> jraph.GraphsTuple:
  """GAE network definition."""
  graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
  
  @jraph.concatenated_args
  def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    net = hk.Sequential([hk.Linear(hidden_dim), jax.nn.relu, hk.Linear(latent_dim)])
    return net(feats)
  
  net = jraph.GraphConvolution(
    update_node_fn=node_update_fn, 
    add_self_edges=True)
  return net(graph)


def inner_product_decode(pred_graph_nodes: jnp.ndarray, senders: jnp.ndarray,
           receivers: jnp.ndarray) -> jnp.ndarray:
  """Given a set of candidate edges, take dot product of respective nodes.

  Args:
    pred_graph_nodes: input graph nodes Z.
    senders: Senders of candidate edges.
    receivers: Receivers of candidate edges.

  Returns:
    For each edge, computes dot product of the features of the two nodes.

  """
  return jnp.squeeze(
      jnp.sum(pred_graph_nodes[senders] * pred_graph_nodes[receivers], axis=1))