"""Deterministic client sampling policies."""


def uniform(num_clients, n_sampled, rng):
    """Sample without replacement and return sorted client IDs."""
    return sorted(rng.choice(num_clients, n_sampled, replace=False).tolist())


def force_rare_client(num_clients, n_sampled, rng, client_info):
    rare_pool = [c for c in range(num_clients) if client_info[c]["rare_fraction"] > 0]
    if not rare_pool:
        return uniform(num_clients, n_sampled, rng)
    forced = int(rng.choice(rare_pool))
    remaining = n_sampled - 1
    pool = [c for c in range(num_clients) if c != forced]
    extras = rng.choice(pool, remaining, replace=False).tolist() if remaining > 0 else []
    return sorted([forced] + extras)


def sample_clients(num_clients, sample_ratio, rng, client_info=None,
                   force_rare_client=False):
    if sample_ratio >= 1.0:
        return list(range(num_clients))
    n_sampled = max(1, int(num_clients * sample_ratio))
    if force_rare_client and client_info is not None:
        return globals()["force_rare_client"](num_clients, n_sampled, rng, client_info)
    return uniform(num_clients, n_sampled, rng)
