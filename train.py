import haiku as hk
import jax
import jax.numpy as jnp
import optax
from sklearn.metrics import roc_auc_score
from absl import app, flags, logging

import functools
from typing import List, Dict, Any

from dataset import load_dataset, train_val_test_split_edges, negative_sampling
from loss import compute_vgae_loss, compute_gae_loss
from model import gae_encoder, vgae_encoder

flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate for the optimizer.')
flags.DEFINE_integer('epochs', 200, 'Number of training epochs.')
flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension in the AE.')
flags.DEFINE_integer('latent_dim', 16, 'Latent dimension in the AE.')
flags.DEFINE_integer('random_seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_frequency', 10, 'How often to evaluate the model.')
flags.DEFINE_bool('is_vgae', True, 'Using Variational GAE vs vanilla GAE.')
FLAGS = flags.FLAGS

def compute_roc_auc_score(preds: jnp.ndarray,
                          labels: jnp.ndarray) -> jnp.ndarray:
  """Computes roc auc (area under the curve) score for classification."""
  s = jax.nn.sigmoid(preds)
  roc_auc = roc_auc_score(labels, s)
  return roc_auc


def train(dataset: List[Dict[str, Any]]) -> hk.Params:
  """Training loop."""
  key = jax.random.PRNGKey(FLAGS.random_seed)
  # Transform impure network to pure functions with hk.transform.
  net_fn = vgae_encoder if FLAGS.is_vgae else gae_encoder
  net_fn = functools.partial(
    net_fn, hidden_dim=FLAGS.hidden_dim, latent_dim=FLAGS.latent_dim)
  net = hk.without_apply_rng(hk.transform(net_fn))
  
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']
  train_graph, _, val_pos_s, val_pos_r, val_neg_s, val_neg_r, test_pos_s, \
      test_pos_r, test_neg_s, test_neg_r = train_val_test_split_edges(
      graph)

  # Prepare the validation and test data.
  val_senders = jnp.concatenate((val_pos_s, val_neg_s))
  val_receivers = jnp.concatenate((val_pos_r, val_neg_r))
  val_labels = jnp.concatenate(
      (jnp.ones(len(val_pos_s)), jnp.zeros(len(val_neg_s))))
  test_senders = jnp.concatenate((test_pos_s, test_neg_s))
  test_receivers = jnp.concatenate((test_pos_r, test_neg_r))
  test_labels = jnp.concatenate(
      (jnp.ones(len(test_pos_s)), jnp.zeros(len(test_neg_s))))
  # Initialize the network.
  key, param_key = jax.random.split(key)
  params = net.init(param_key, train_graph)
  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(FLAGS.learning_rate)
  opt_state = opt_init(params)
  
  if FLAGS.is_vgae:
    key, loss_key = jax.random.split(key)
    loss_fn = functools.partial(compute_vgae_loss, rng_key=loss_key)
  else:
    loss_fn = compute_gae_loss
  compute_loss_fn = functools.partial(loss_fn, net=net)
  # We jit the computation of our loss, since this is the main computation.
  compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))

  key, *neg_sampling_keys = jax.random.split(key, FLAGS.epochs+1)
  for epoch in range(FLAGS.epochs):
    num_neg_samples = train_graph.senders.shape[0]
    train_neg_senders, train_neg_receivers = negative_sampling(
        train_graph, num_neg_samples=num_neg_samples, key=neg_sampling_keys[epoch])
    train_senders = jnp.concatenate((train_graph.senders, train_neg_senders))
    train_receivers = jnp.concatenate(
        (train_graph.receivers, train_neg_receivers))
    train_labels = jnp.concatenate(
        (jnp.ones(len(train_graph.senders)), jnp.zeros(len(train_neg_senders))))
  
    (train_loss,
     train_preds), grad = compute_loss_fn(params, train_graph, train_senders,
                                          train_receivers, train_labels)

    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    if epoch % FLAGS.eval_frequency == 0 or epoch == (FLAGS.epochs - 1):
      train_roc_auc = compute_roc_auc_score(train_preds, train_labels)
      val_loss, val_preds = loss_fn(params, train_graph, val_senders,
                                         val_receivers, val_labels, net)
      val_roc_auc = compute_roc_auc_score(val_preds, val_labels)
      logging.info(f'epoch: {epoch}, train_loss: {train_loss:.3f}, '
            f'train_roc_auc: {train_roc_auc:.3f}, val_loss: {val_loss:.3f}, '
            f'val_roc_auc: {val_roc_auc:.3f}')
  test_loss, test_preds = loss_fn(params, train_graph, test_senders,
                                       test_receivers, test_labels, net)
  test_roc_auc = compute_roc_auc_score(test_preds, test_labels)
  logging.info('Training finished')
  logging.info(
      f'epoch: {epoch}, test_loss: {test_loss:.3f}, test_roc_auc: {test_roc_auc:.3f}'
  )
  return params


def main(_):
    cora_ds = load_dataset('./dataset/cora.pickle')
    _ = train(cora_ds)

if __name__ == '__main__':
    app.run(main)