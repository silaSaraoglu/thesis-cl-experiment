"""Paired one-sided permutation tests for comparing models across task-order configs.

Every model in an experiment is trained on the *same* task-order configurations
(config_1 .. config_15 in config.py), so comparing two models is a paired design: the
configs are matched blocks and the between-config variance -- which is large in CL,
since task order drives forgetting -- cancels out of the comparison.

The hypotheses are the ones stated in the thesis protocol, on the population means of
the metric under consideration:

    H0: mu_A <= mu_B        H1: mu_A > mu_B     (one-sided, alpha = 0.05)

tested on the paired differences d_i = A_i - B_i via an exact sign-flip permutation
test. Under H0 the sign of each d_i is exchangeable, so enumerating all 2**n sign
assignments yields the exact null distribution of mean(d); for n = 15 that is 32768
assignments, cheap enough to enumerate in full, so no sampling error enters the
p-value. Above `max_exact` differences the enumeration falls back to random sign draws.

The test statistic is the mean difference, which keeps the test faithful to hypotheses
written about mu -- unlike a signed-rank test, which targets the pseudo-median instead.
"""

import math

import numpy as np

ALPHA = 0.05

# Nemenyi critical values q_alpha (studentized range divided by sqrt(2)), indexed by the
# number of models k, from Demsar (2006) Table 5. Only k >= 2 is meaningful.
_NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
           9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159},
}


def _exact_null_means(d):
    """Every 2**n sign-flipped mean of `d`, enumerated in chunks to bound memory."""
    n = d.size
    total = 1 << n
    out = np.empty(total, dtype=np.float64)
    bits = np.arange(n, dtype=np.int64)
    step = 1 << 12
    for start in range(0, total, step):
        idx = np.arange(start, min(start + step, total), dtype=np.int64)[:, None]
        signs = 1.0 - 2.0 * ((idx >> bits) & 1)
        out[start:start + signs.shape[0]] = signs @ d / n
    return out


def _sampled_null_means(d, n_resamples, rng):
    n = d.size
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    return signs @ d / n


def paired_permutation_greater(a, b, max_exact=20, n_resamples=100_000, seed=0):
    """One-sided paired permutation test of H1: mean(a) > mean(b).

    Returns a dict with the p-value, the observed mean difference, the number of
    configs on which `a` wins, and whether the null distribution was enumerated
    exactly. `a` and `b` must be matched element-wise (same config in each position).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test needs equal-length samples, got %s vs %s"
                         % (a.shape, b.shape))
    d = a - b
    n = d.size
    observed = float(d.mean())

    exact = n <= max_exact
    if exact:
        null = _exact_null_means(d)
        # The all-(+1) assignment reproduces `observed`, so it is counted here and the
        # p-value can never be 0 -- the exact floor is 1 / 2**n (3.05e-05 at n = 15).
        p = float(np.count_nonzero(null >= observed - 1e-12) / null.size)
    else:
        null = _sampled_null_means(d, n_resamples, np.random.default_rng(seed))
        p = float((1 + np.count_nonzero(null >= observed - 1e-12)) / (n_resamples + 1))

    return {
        "p_value":   p,
        "mean_a":    float(a.mean()),
        "mean_b":    float(b.mean()),
        "mean_diff": observed,
        "wins":      int(np.count_nonzero(d > 0)),
        "ties":      int(np.count_nonzero(d == 0)),
        "n":         n,
        "exact":     exact,
    }


def holm_adjust(p_values):
    """Holm step-down adjusted p-values (compare against alpha, as usual).

    Controls the family-wise error rate: the chance that *any* comparison in the family
    is a false positive stays at alpha, rather than each comparison separately having an
    alpha chance. Uniformly more powerful than Bonferroni at the same guarantee.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adjusted = np.maximum.accumulate(scaled)  # keep the sequence non-decreasing
    out = np.empty(m, dtype=np.float64)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def pairwise_one_sided(values_by_model, alpha=ALPHA, **kwargs):
    """Test every ordered pair (A beats B?) and Holm-correct within the whole family.

    `values_by_model` maps model name -> sequence of per-config metric values, all in
    the same config order. Returns a list of records, one per ordered pair.

    The correction family is every ordered pair actually tested. Testing both
    directions of a pair is what makes this necessary in the first place: each
    direction at alpha would give the pair a 2*alpha chance of declaring some winner.
    """
    names = list(values_by_model)
    records = []
    for a in names:
        for b in names:
            if a == b:
                continue
            res = paired_permutation_greater(values_by_model[a], values_by_model[b],
                                             **kwargs)
            res["model_a"] = a
            res["model_b"] = b
            records.append(res)

    adjusted = holm_adjust([r["p_value"] for r in records])
    for rec, p_adj in zip(records, adjusted):
        rec["p_holm"] = float(p_adj)
        rec["significant_raw"] = rec["p_value"] < alpha
        rec["significant_holm"] = p_adj < alpha
    return records


# --------------------------------------------------------------------------------
# Friedman omnibus test + Nemenyi post-hoc
#
# Where the pairwise tests above compare two models at a time on their actual metric
# values, this pair of tests looks at all k models at once through their *ranks* within
# each block (here: each task-order config). Friedman asks the single global question
# "are the average ranks more spread out than chance allows?"; only if it rejects does
# the Nemenyi post-hoc identify which models differ, via one shared Critical Difference
# threshold on the average ranks. Ranks discard magnitude, so this is the more
# conservative and less informative of the two views -- but it needs no correction
# bookkeeping and summarizes into a single CD diagram.
# --------------------------------------------------------------------------------


def _chi2_sf(x, df):
    """P(X > x) for a chi-square with integer df, via the standard Q(a, y) recurrence.

    Exact for the integer df used here (df = k - 1), so the module keeps its numpy-only
    dependency rather than pulling in scipy for one distribution tail.
    """
    if x <= 0:
        return 1.0
    y = x / 2.0
    if df % 2 == 0:
        a, q = 1.0, math.exp(-y)          # Q(1, y)
    else:
        a, q = 0.5, math.erfc(math.sqrt(y))  # Q(1/2, y)
    # Q(a + 1, y) = Q(a, y) + y**a * exp(-y) / Gamma(a + 1)
    while a < df / 2.0:
        q += y ** a * math.exp(-y) / math.gamma(a + 1.0)
        a += 1.0
    return float(min(max(q, 0.0), 1.0))


def average_ranks(matrix):
    """Mean rank of each model across blocks; rank 1 = best (highest metric).

    Ties inside a block share the average of the ranks they span, the standard
    correction so that every block's ranks still sum to k(k + 1) / 2.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n_blocks, k = matrix.shape
    ranks = np.empty_like(matrix)
    for i in range(n_blocks):
        order = np.argsort(-matrix[i], kind="stable")
        raw = np.empty(k, dtype=np.float64)
        raw[order] = np.arange(1, k + 1, dtype=np.float64)
        # average the ranks of tied values
        for value in np.unique(matrix[i]):
            tied = matrix[i] == value
            if tied.sum() > 1:
                raw[tied] = raw[tied].mean()
        ranks[i] = raw
    return ranks.mean(axis=0)


def friedman_test(matrix):
    """Friedman omnibus test over a blocks x models matrix of metric values.

    H0: all k models have the same expected rank. Non-parametric counterpart of a
    repeated-measures ANOVA, and it assumes exactly the dependent sampling this data
    has -- every model is measured on every block.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n, k = matrix.shape
    if k < 3:
        raise ValueError("Friedman needs at least 3 models, got %d" % k)
    mean_ranks = average_ranks(matrix)
    statistic = (12.0 * n / (k * (k + 1))
                 * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4.0))
    df = k - 1
    return {
        "statistic":    float(statistic),
        "df":           df,
        "p_value":      _chi2_sf(statistic, df),
        "average_ranks": mean_ranks,
        "n_blocks":     int(n),
        "k_models":     int(k),
    }


def nemenyi_cd(k, n, alpha=ALPHA):
    """Critical Difference: the smallest average-rank gap that counts as significant.

    Two models differ significantly at `alpha` iff their average ranks differ by more
    than this. Widens with more models and narrows with more blocks.
    """
    table = _NEMENYI_Q.get(alpha)
    if table is None:
        raise ValueError("no Nemenyi critical values tabulated for alpha=%g "
                         "(have %s)" % (alpha, sorted(_NEMENYI_Q)))
    if k not in table:
        raise ValueError("no Nemenyi critical value for k=%d models (have 2..%d)"
                         % (k, max(table)))
    return float(table[k] * math.sqrt(k * (k + 1) / (6.0 * n)))

