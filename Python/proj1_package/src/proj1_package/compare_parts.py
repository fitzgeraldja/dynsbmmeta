import numpy as np
import numpy.ma as ma

from itertools import permutations as perms


def max_global_overlap_over_perms(Z_1, Z_2, Q):
    """
    Calculate maximum overlap of two dynamic partitions, for a single global label permutation

    Args:
        Z_1 (T x N np.array): First temporal partition
        Z_2 (T x N np.array): Second temporal partition
        Q (int): Number of communities (could be inferred as max over Z_1 and Z_2)

    Returns:
        best_perm (list length Q): Best permutation of labels found
        overlaps (np.array length T): overlap at each timestep given this permutation
    """
    vals, idxs = np.unique(Z_1, return_inverse=True)

    if -1 in vals:
        # must account for missing nodes
        # mask non-present nodes for fair comparison, assuming this has code -1
        mZ_2 = ma.masked_equal(Z_2, -1)
        best_perm = max(
            perms(range(Q)),
            key=lambda perm: (
                np.array([-1] + list(perm))[idxs].reshape(Z_1.shape) == mZ_2
            ).sum(),
        )
        overlaps = (
            np.array([-1] + list(best_perm))[idxs].reshape(Z_1.shape) == mZ_2
        ).sum(axis=1).astype(np.double) / (Z_1 != -1).sum(axis=0)
    else:
        best_perm = max(
            perms(range(Q)),
            key=lambda perm: (np.array(perm)[idxs].reshape(Z_1.shape) == Z_2).sum(),
        )
        overlaps = (np.array(best_perm)[idxs].reshape(Z_1.shape) == Z_2).sum(
            axis=1
        ).astype(np.double) / (Z_1.shape[1])

    return best_perm, overlaps


def max_local_overlap_over_perms(Z_1, Z_2, Q):
    """
    Calculate maximum overlap of two dynamic partitions, for a local label permutation within each timestep
    - note not a good measure, but how generally done in literature.

    Args:
        Z_1 (T x N np.array): Predicted temporal partition
        Z_2 (T x N np.array): True temporal partition
        Q (int): Number of communities (could be inferred as max over Z_1 and Z_2)

    Returns:
        best_perms (T x Q np.array): Best permutation of labels found at each timestep
        overlaps (np.array length T): overlap at each timestep given this permutation
    """
    overlaps = np.zeros((len(Z_1),))
    best_perms = np.zeros((len(Z_1), Q))
    for t in range(len(Z_1)):
        vals, idxs = np.unique(Z_1[t, :], return_inverse=True)
        if -1 not in vals:
            best_perm = max(
                perms(range(Q)),
                key=lambda perm: (
                    np.array(perm)[idxs].reshape(Z_1.shape[1]) == Z_2[t, :]
                ).sum(),
            )
            overlaps[t] = (
                np.array(best_perm)[idxs].reshape(Z_1.shape[1]) == Z_2[t, :]
            ).sum().astype(np.double) / Z_1.shape[1]
            best_perms[t, :] = best_perm
        else:
            mZ_2 = ma.masked_equal(Z_2, -1)
            best_perm = max(
                perms(range(Q)),
                key=lambda perm: (
                    np.array([-1] + list(perm))[idxs].reshape(Z_1.shape[1])
                    == mZ_2[t, :]
                ).sum(),
            )
            overlaps[t] = (
                np.array([-1] + list(best_perm))[idxs].reshape(Z_1.shape[1])
                == mZ_2[t, :]
            ).sum().astype(np.double) / (Z_1[t, :] != -1).sum()
            best_perms[t, :] = best_perm
    return best_perms, overlaps


def overlap_given_perm(Z_1, Z_2, perm):
    """
    Calculate overlaps for two temporal partitions given a permutation of labels

    Args:
        Z_1 ([type]): [description]
        Z_2 ([type]): [description]
        perm ([type]): [description]
    """
    overlaps = np.zeros((len(Z_1),))
    for t in range(len(Z_1)):
        _, idxs = np.unique(Z_1[t, :], return_inverse=True)
        overlaps[t] = (
            np.array(perm)[idxs].reshape(Z_1.shape[1]) == Z_2[t, :]
        ).sum().astype(np.double) / Z_1.shape[1]
    return overlaps


def permute_labels_given_perm(Z, perm):
    """
    Permute labels of a temporal partition

    Args:
        Z ([type]): [description]
        perm ([type]): [description]
    """
    nZ = Z.copy()
    for t in range(nZ.shape[0]):
        _, idxs = np.unique(nZ[t, :], return_inverse=True)
        nZ[t, :] = np.array(perm)[idxs]
    return nZ


def best_global_perm_from_mse(M_1, M_2):
    """
    Find best shared permutation of rows and columns to minimise MSE between two square matrices of same dimensions

    Args:
        M_1 ([type]): T x Q x Q array to permute within each timeslice
        M_2 ([type]): T x Q x Q array against which to test
    """
    Q = M_1.shape[1]
    best_perm = min(
        perms(range(Q)),
        key=lambda perm: np.mean(np.square(temporal_matrix_perm(M_1, perm) - M_2)),
    )
    # print(f"{sorted([np.mean(np.square(matrix_perm(M_1,perm)-M_2)) for perm in perms(range(Q))])}")
    return best_perm


def temporal_matrix_perm(M, perm):
    """
    Permute rows and columns of square matrix M according to given permutation vector

    Args:
        M ([type]): [description]
        perm ([type]): [description]
    """
    nM = M.copy()
    for t in range(M.shape[0]):
        nM[t, :, :] = nM[t, :, perm]
        nM[t, :, :] = nM[t, perm, :]
    return nM


def calc_flexibility(Z, i=None, method="base"):
    """
    Calculate flexibility of all/given node(s), defined as the ratio of the number of observed group changes to the number possible (i.e. number of timesteps).
    Note at next level, should account for similarity of communities, i.e. a change is more significant if it is to a community that is more diferent/less connected to
    the prior community. This could be measured by e.g.
        (i) the inferred likelihood of making this transition (i.e. pi_qq'),
        (ii) the previous likelihood of connections to q' (i.e. beta^{t-1}_qq')
        (iii)

    Args:
        Z (T x N np.array): Temporal partition
        i (int, optional): Specific node index to calculate flexibilty for. Defaults to None, in which case all flexibilities calculated.
    """
    T = Z.shape[0]
    if method == "base":
        base_flex = (
            np.hstack([Z[t - 1, :] != Z[t, :] for t in range(1, T)]).sum(axis=1) / T
        )
        return base_flex


def eff_no_groups(part):
    """
    Calculate effective number of groups of partition, i.e. exponential of entropy of group size distribution.

    Args:
        part ([type]): Single dimensional partition (i.e. within a single timeslice)
    """
    # discard non-present nodes
    part = part[part != -1]
    _, counts = np.unique(part, return_counts=True)
    freqs = counts / counts.sum()
    return np.exp(-np.multiply(freqs, np.log(freqs)).sum())


def jacc_idx(Z):
    """
    Calculate Jaccard index for group transitions over time. Note if Q and/or N gets particularly large, much better to just use minhash rather than exact

    Args:
        Z ([type]): [description]

    """
    T = Z.shape[0]
    Q = np.amax(Z) + 1
    jaccs = np.array(
        [
            [
                [
                    len(
                        set(np.nonzero(Z[t] == q)[0])
                        & set(np.nonzero(Z[t + 1] == r)[0])
                    )
                    / len(
                        set(np.nonzero(Z[t] == q)[0])
                        | set(np.nonzero(Z[t + 1] == r)[0])
                    )
                    for r in range(Q)
                ]
                for q in range(Q)
            ]
            for t in range(T - 1)
        ]
    )
    return jaccs


def jacc_flex(Z):
    """
    Calculate flexibility of each author as defined by average jaccard index of their inferred group changes

    Args:
        Z ([type]): [description]

    Returns:
        [type]: [description]
    """
    jaccs = jacc_idx(Z)
    T = Z.shape[0]
    jacc_flex = np.array(
        [
            np.nanmean(
                [
                    jaccs[t, Z[t, i], Z[t + 1, i]]
                    if (Z[t, i] != -1 and Z[t + 1, i] != -1)
                    else np.nan
                    for t in range(T - 1)
                ]
            )
            for i in range(Z.shape[1])
        ]
    )

    return jacc_flex


def tau_norm_entropy(taum_vals):
    """
    Calculate entropy of group marginals inferred at each timestep for each node

    Args:
        taum_vals ([type]): [description]

    Returns:
        [type]: [description]
    """
    Q = taum_vals.shape[-1]
    norm_ents = np.einsum("...k,...k", -taum_vals, np.log(taum_vals))
    norm_ents = norm_ents / np.log(Q)
    return norm_ents
