import itertools

import numpy as np
import utils.my_logging as my_logging


class NDDistribution:

    def __init__(
        self,
        data: dict,
        bins_arg=50,
    ):

        self.logger = my_logging.get_logger("NDDistribution")
        self.data = data
        self.ndim = len(data)
        # Assert all data arrays have the same length
        if not all(len(v) == len(next(iter(data.values()))) for v in data.values()):
            raise ValueError("All data arrays must have the same length.")
        self.npoints = len(next(iter(data.values())))
        self.bins_arg = bins_arg
        self.hist = self.bins = self.sorted_idxs = self.cdf = self.hist_flat = None
        self.logger.info(
            f"Creating NDDistribution with {self.npoints:_} data points of {self.ndim}."
        )
        self.logger.info("Data keys: " + ", ".join(data.keys()))
        self.set_distribution()
        self.nbins = len(self.bins[0])

        pass

    @classmethod
    def from_pandas(cls, df, names=None, **kwargs):
        """
        Create an NDDistribution instance from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        names : list of str, optional
            The names of the columns to use as data, by default None.
            If None, all columns are used.
        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        NDDistribution
            An instance of NDDistribution.
        """
        if names is None:
            names = df.columns.tolist()
        data = {name: df[name].values for name in names}
        return cls(data, **kwargs)

    @classmethod
    def compute_distribution(cls, data, bins_arg):
        hist, bins = np.histogramdd(
            np.stack(list(data.values())).T, bins=bins_arg, density=True
        )
        return hist, bins

    def set_distribution(self):
        """
        Set the histogram and bins attributes based on the computed distribution.
        """
        self.logger.info(f"Computing {self.ndim}-dimensional distribution...")
        self.hist, bins = self.compute_distribution(self.data, self.bins_arg)
        self.bins = np.stack(bins)
        self.hist_flat = self.hist.ravel()
        self.sorted_idxs = np.argsort(self.hist_flat)
        cdf = np.cumsum(self.hist_flat[self.sorted_idxs])
        self.cdf = cdf / cdf[-1]

    def sample(self, num_samples: int):
        if self.hist is None or self.bins is None:
            raise ValueError(
                "Distribution not computed. Call compute_distribution() first."
            )

        # Get bin centers and widths
        centers = 0.5 * (self.bins[:, :-1] + self.bins[:, 1:])
        bin_width = centers[:, 1] - centers[:, 0]
        # Create flat grid of bin centers (i.e. coordinate grid in sample space)
        combinations = np.array(list(itertools.product(*centers)))
        # Sample bin centers
        samples = combinations[
            np.random.choice(
                combinations.shape[0],
                num_samples,
                p=self.hist.ravel() / self.hist.sum(),
            )
        ]
        # Sample uniformly within bin
        samples += np.random.uniform(-0.5, 0.5, size=samples.shape) * bin_width
        return samples

    def get_percentile(self, samples):
        """
        For an array of samples, return the percentile of each sample based on
        the distribution defined by the histogram and bins.


        Parameters
        ----------
        samples : float, array-like
            Samples to compute percentiles for.

        Returns
        -------
        percentiles : float, array-like
            Percentiles of the samples based on the distribution.
        """
        samples = np.atleast_2d(samples)

        # Find out which histogram bin each sample falls into
        bin_idxs = np.array(
            [
                np.digitize(samples[:, i], self.bins[i][:-1], right=False)
                for i in range(self.ndim)
            ]
        )

        # Get hist values of the bins where the samples fall into
        hist_values = self.hist[tuple(np.clip(bin_idxs, 0, self.nbins - 1))]

        # Find percentiles for each sample based on where it falls in the
        # cumulative distribution function (CDF)
        percentiles = self.cdf[np.searchsorted(self.cdf, hist_values)]
        return percentiles
