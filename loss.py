import jax.numpy as jnp

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


def compute_kl_gaussian(mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate KL divergence between given and standard gaussian distributions.

    Args:
        mean: feature matrix of the mean.
        log_std: feature matrix of the log-covariance.

    Returns:
        A vector representing KL divergence of the two Gaussian distributions
        of length |V| where V is the nodes in the graph.
    """
    var = jnp.exp(log_std)
    return 0.5 * jnp.sum(
      -2*log_std - 1.0 + jnp.square(var) + jnp.square(mean), axis=-1)
  
  