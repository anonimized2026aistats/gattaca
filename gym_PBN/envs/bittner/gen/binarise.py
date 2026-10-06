import math

import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.cluster import KMeans


def binarise(df: pd.DataFrame, method: str) -> pd.DataFrame:
    function_map = {
        "average": lambda x: _threshold(x, lambda y: y.T.mean()),
        "median": lambda x: _threshold(x, lambda y: y.T.median()),
        # "kmeans": lambda x: x.apply(_binarise_gene_kmeans, axis=1),
        "kmeans": lambda x: _threshold(
            x, lambda y: KMeansLegacyV2().fit_thresholds(y.T)
        ),
    }

    if method not in function_map:
        raise Exception(f'"{method}" is not an implemented binarisation method.')

    names = df["Name"]
    data = df.drop("Name", axis=1)

    binarised = function_map[method](data)
    binarised["Name"] = names

    return binarised


def _threshold(x, compute_thresh_func):
    threshold = compute_thresh_func(x)
    return (x.T > threshold).astype(int).T


def _binarise_gene_kmeans(x):
    """New version of KMeans binarisation using sklearn. Currently unused."""
    # TODO maybe re-introduce the evaluation step
    assert isinstance(x, pd.Series)

    x_log = np.log(x)
    kmeans = KMeans(n_clusters=2).fit(x_log.to_numpy().reshape(-1, 1))

    centres = [i[0] for i in kmeans.cluster_centers_]
    labels = kmeans.labels_

    if centres[0] > centres[1]:  # Assert that cluster 0 is the lower expression level
        labels = abs(kmeans.labels_ - 1)

    ret = x.copy()
    for i in range(len(ret.index)):
        ret.loc[x.index[i]] = labels[i]

    return ret.astype(int)


class KMeansLegacy:
    """KMeans binarisation method written by the previous code owner.

    It has a lot of problems:
    - Why do we binarise based on thresholds and not based on the cluster assignment from KMeans?
      - In theory this shouldn't matter if the thresholds are calculated properly.
    - Why do we evaluate how well a set of clusters fits the entire dataset?
      - Each run, we fit KMeans clusters to each individual gene to get its custom threshold.
      - But then we aggregate all the different clusters for each run and evaluate the run as a whole.
      - A run might be god tier for some genes but bad for others and be discarded for no reason.
      - Perhaps we should do individual runs per-gene and evaluate per-gene.
    - KMeans hyper-params are kinda weird. sklearn defaults are 10 runs 300 iters.
    - Just where did the evaluation criterion come from???

    Keeping it for legacy reasons and because the evaluation criterion might actually be better than
    default KMeans from sklearn, I wouldn't know.
    """

    def __init__(self, n_clusters=2, n_init=10, max_iter=20) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter

    def fit_thresholds(self, x):
        x_log = np.log1p(x)

        # Creating clusters n_runs times
        thresholds = [self.cluster(x_log) for _ in range(self.n_init)]

        # Evaluating clusters to pick the best run
        threshold = thresholds[
            np.argmin([self.eval_cluster(x_log, threshold) for threshold in thresholds])
        ]

        # HACK: binary only
        output = pd.DataFrame(threshold)[0].apply(lambda y: np.expm1(y[0]))
        output.index = x.columns
        return output

    def _cluster_gene(self, exp):
        min_val = exp.min()
        max_val = exp.max()
        means = np.random.rand(self.n_clusters) * (max_val - min_val) + min_val

        for i in range(self.max_iter):
            clusters = [[]] * self.n_clusters

            for val in exp:
                cluster = np.argmin([abs(val - mean) for mean in means])
                clusters[cluster].append(val)

            means = np.mean(clusters, axis=1)

        for i in range(len(clusters)):
            clusters[i].sort()

        cluster_sort_idx = np.argsort(means)
        t = np.empty(self.n_clusters - 1)

        for i in range(cluster_sort_idx.size - 1):
            r1 = clusters[cluster_sort_idx[i]][-1]  # max value
            r2 = clusters[cluster_sort_idx[i + 1]][0]  # min value
            t[i] = (r1 + r2) / 2  # threshold is cluster mean

        return t, means[cluster_sort_idx]

    def cluster(self, x):
        thresholds = []

        for k in range(x.shape[1]):
            exp = np.array(x).take(k, axis=1)
            thresholds.append(self._cluster_gene(exp))

        return thresholds

    @classmethod
    def _eval_cluster_gene(cls, exp, threshold):
        (t, means) = threshold

        # Parameters for the gaussian.
        p_mean = np.mean(exp)
        std = np.std(exp)

        errors = np.zeros(len(means))

        # HACK: 2 clusters only
        clusters = [[]] * 2
        for val in exp:
            clusters[int(val >= t)].append(val)

        for i in range(len(clusters)):
            if i == 0:
                t_lower = min(clusters[i])
                t_upper = t[0]
            else:
                t_lower = t[0]
                t_upper = max(clusters[i])
            I = quad(cls._integrand, t_lower, t_upper, args=(p_mean, std, means[i]))
            errors[i] = I[0]

        return np.sum(errors)

    @classmethod
    def eval_cluster(cls, x, thresholds):
        e = 0
        for j in range(x.shape[1]):
            exp = np.array(x).take(j, axis=1)
            e += cls._eval_cluster_gene(exp, thresholds[j])
        return e

    @classmethod
    def _gaussian(cls, x, mean, std):
        a = 1 / (std * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((x - mean) / std) ** 2
        output = a * math.exp(exponent)
        return output

    @classmethod
    def _integrand(cls, x, mean, std, r):
        return ((x - r) ** 2) * cls._gaussian(x, mean, std)


class KMeansLegacyV2(KMeansLegacy):
    """Upgraded version of KMeansLegacy to do clustering optimisation for each gene individually."""

    def fit_thresholds(self, x):
        x_log = np.log1p(x)
        ret = []

        # Creating clusters n_runs times
        for i in range(x.shape[1]):
            # fmt: off
            exp = np.array(x_log).take(i, axis=1)
            thresholds = [self._cluster_gene(exp) for _ in range(self.n_init)]
            threshold = thresholds[
                np.argmin([self._eval_cluster_gene(exp, threshold) for threshold in thresholds])
            ]
            # fmt: on

            # HACK: Binary only
            ret.append(np.expm1(threshold[0][0]))

        ret = pd.Series(ret, index=x.columns)
        return ret
