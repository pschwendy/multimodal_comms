"""Autoencoder harvesting/pretraining hooks built around injected trainer objects."""

from .toy import LinearTrainer, TrainingBatch


def train_one_batch(trainer: LinearTrainer, batch: TrainingBatch) -> float:
    return trainer.step(batch)


def clone_decode_embeddings(embed_layer, decode_ids):
    """Return writable decode embeddings without mutating a frozen embedding view.

    The `.clone()` keeps sampled-latent and packed-autoencoder training from
    writing through a frozen embedding view. Keeping it in one workflow helper makes the
    required gradient boundary explicit for real torch modules and test doubles.
    """
    embeddings = embed_layer(decode_ids)
    if not hasattr(embeddings, "clone"):
        raise TypeError("embedding result must provide clone()")
    return embeddings.clone()
