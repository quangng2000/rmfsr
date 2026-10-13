"""JVP and velocity target extracted from the MeanFlow authors' PyTorch code.

Source: Gsunshine/py-meanflow, commit
1f6d72d94247c8fdeb89489acca1a8007a6baf6c,
meanflow/models/meanflow.py, MeanFlow.forward_with_loss, lines 51-57.
Original and modifications: CC-BY-NC-SA-4.0; see LICENSE and NOTICE.md.

Modifications: expose the network callback and precomputed path as arguments;
allow an explicit spatial tangent override for the speech IMF formulation;
disable autocast on the input device instead of hard-coding CUDA. The default
tangent and detached target are the original MeanFlow equations. No speech,
network architecture, EMA, time sampling or loss reduction belongs here.
"""

import torch


def velocity_target(u_func, z, t, r, v, *, tangent=None):
    """Return ``(predicted_velocity, detached_target_velocity)``.

    ``u_func(z, t, r)`` predicts interval-average velocity. ``v`` is the
    conditional path velocity; it is also the JVP tangent unless overridden.
    Times must be shaped to broadcast over ``z`` (upstream uses B x 1 x 1 x 1).
    The caller owns the data path, time interval, and loss weighting.
    """
    dtdt = torch.ones_like(t)
    drdt = torch.zeros_like(r)

    with torch.amp.autocast(z.device.type, enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t, r), (v if tangent is None else tangent, dtdt, drdt))
        u_tgt = (v - (t - r) * dudt).detach()

    return u_pred, u_tgt
