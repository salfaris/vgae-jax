import jax
import jax.numpy as jnp
import haiku as hk
import jraph
from sklearn.metrics import roc_auc_score

from typing import Tuple

def compute_bce_with_logits_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes binary cross-entropy with logits loss.

  Combines sigmoid and BCE, and uses log-sum-exp trick for numerical stability.
  See https://stackoverflow.com/a/66909858 if you want to learn more.

  Args:
    x: Predictions (logits).
    y: Labels.

  Returns:
    Binary cross-entropy loss with mean aggregation.

  """
  max_val = jnp.clip(-x, 0, None)
  loss = x - x*y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x-max_val)))
  return jnp.mean(loss, axis=-1)

def compute_weighted_bce_with_logits_loss(
  x: jnp.ndarray, y: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
  """Computes weighted binary cross-entropy with logits loss.

  Combines sigmoid and BCE, and uses log-sum-exp trick for numerical stability.
  See https://stackoverflow.com/a/66909858 if you want to learn more.

  Args:
    x: Predictions (logits).
    y: Labels.

  Returns:
    Binary cross-entropy loss.

  """
  max_val = jnp.clip(-x, 0, None)
  loss = x - x*y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x-max_val)))
  loss = weight * loss
  return jnp.mean(loss, axis=-1)


def compute_kl_gaussian(mean: jnp.ndarray, log_var: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate KL divergence between given and standard gaussian distributions.

    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)

    Args:
        mean: mean vector of the first distribution
        log_var: diagonal vector of log-covariance matrix of the first distribution

    Returns:
        A scalar representing KL divergence of the two Gaussian distributions.
    """
    return 0.5 * jnp.sum(-log_var - 1.0 + jnp.exp(log_var) + jnp.square(mean), axis=-1)
  
  