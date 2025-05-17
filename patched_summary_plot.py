""" Adapted from https://github.com/shap/shap/blob/master/shap/plots/_beeswarm.py """

""" Summary plots of SHAP values across a whole dataset.
"""

from __future__ import division

import warnings

import numpy as np
import scipy.cluster
import scipy.sparse
import scipy.spatial
from scipy.stats import gaussian_kde

try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from shap import Explanation
from shap.utils import safe_isinstance
from shap.utils._exceptions import DimensionError
from shap.plots import colors
from shap.plots._labels import labels
from shap.plots._utils import (
    convert_color,
    convert_ordering,
    get_sort_order,
    merge_nodes,
    sort_inds,
)


def summary_legacy(
        shap_values,
        features=None,
        feature_names=None,
        feature_order=None,
        max_display=None,
        plot_type=None,
        color=None,
        axis_color="#333333",
        title=None,
        alpha=1,
        show=True,
        sort=True,
        color_bar=True,
        plot_size="auto",
        layered_violin_max_num_bins=20,
        class_names=None,
        class_inds=None,
        color_bar_label=labels["FEATURE_VALUE"],
        cmap=colors.red_blue,
        show_values_in_legend=False,
        use_log_scale=False,
):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.

    show_values_in_legend: bool
        Flag to print the mean of the SHAP values in the multi-output bar plot. Set to False
        by default.

    """
    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names
        # if out_names is None: # TODO: waiting for slicer support of this
        #     out_names = shap_exp.output_names

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar"  # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot"  # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == "layered_violin":
            color = "coolwarm"
        elif multi_class:

            def color(i):
                return colors.red_blue_circle(i / len(shap_values))
        else:
            color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = shap_values[0].shape[1] if multi_class else shap_values.shape[1]

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, (
                    shape_msg + " Perhaps the extra column in the shap_values matrix is the "
                                "constant offset? Of so just pass shap_values[:,:-1]."
            )
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    if use_log_scale:
        pl.xscale("symlog")

    # plotting SHAP interaction values
    if not multi_class and len(shap_values.shape) == 3:
        if plot_type == "compact_dot":
            new_shap_values = shap_values.reshape(shap_values.shape[0], -1)
            new_features = np.tile(features, (1, 1, features.shape[1])).reshape(features.shape[0], -1)

            new_feature_names = []
            for c1 in feature_names:
                for c2 in feature_names:
                    if c1 == c2:
                        new_feature_names.append(c1)
                    else:
                        new_feature_names.append(c1 + "* - " + c2)

            return summary_legacy(
                new_shap_values,
                new_features,
                new_feature_names,
                max_display=max_display,
                plot_type="dot",
                color=color,
                axis_color=axis_color,
                title=title,
                alpha=alpha,
                show=show,
                sort=sort,
                color_bar=color_bar,
                plot_size=plot_size,
                class_names=class_names,
                color_bar_label="*" + color_bar_label,
            )

        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        sort_inds = np.argsort(-np.abs(shap_values.sum(1)).sum(0))

        # get plotting limits
        delta = 1.0 / (shap_values.shape[1] ** 2)
        slow = np.nanpercentile(shap_values, delta)
        shigh = np.nanpercentile(shap_values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        pl.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
        pl.subplot(1, max_display, 1)
        proj_shap_values = shap_values[:, sort_inds[0], sort_inds]
        proj_shap_values[:, 1:] *= 2  # because off diag effects are split in half
        summary_legacy(
            proj_shap_values,
            features[:, sort_inds] if features is not None else None,
            feature_names=np.array(feature_names)[sort_inds].tolist(),
            sort=False,
            show=False,
            color_bar=False,
            plot_size=None,
            max_display=max_display,
        )
        pl.xlim((slow, shigh))
        pl.xlabel("")
        title_length_limit = 11
        pl.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))
        for i in range(1, min(len(sort_inds), max_display)):
            ind = sort_inds[i]
            pl.subplot(1, max_display, i + 1)
            proj_shap_values = shap_values[:, ind, sort_inds]
            proj_shap_values *= 2
            proj_shap_values[:, i] /= 2  # because only off diag effects are split in half
            summary_legacy(
                proj_shap_values,
                features[:, sort_inds] if features is not None else None,
                sort=False,
                feature_names=["" for i in range(len(feature_names))],
                show=False,
                color_bar=False,
                plot_size=None,
                max_display=max_display,
            )
            pl.xlim((slow, shigh))
            pl.xlabel("")
            if i == min(len(sort_inds), max_display) // 2:
                pl.xlabel(labels["INTERACTION_VALUE"])
            pl.title(shorten_text(feature_names[ind], title_length_limit))
        pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        pl.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            pl.show()
        return

    if max_display is None:
        max_display = 20

    if not feature_order:
        if sort:
            # order features by the sum of their effect magnitudes
            if multi_class:
                feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
            else:
                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
            feature_order = feature_order[-min(max_display, len(feature_order)):]
        else:
            feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    print(feature_order)

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]

            rng = np.random.default_rng(i)
            zero_mask = values > 0.001
            rnd_mask = rng.integers(0, high=2, size=zero_mask.shape, dtype=int).astype(bool)
            for _ in range(4):
                rnd_mask = rnd_mask & rng.integers(0, high=2, size=zero_mask.shape, dtype=int).astype(bool)
            select_mask = zero_mask | rnd_mask
            shaps = shaps[select_mask]
            values = values[select_mask]

            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                if idx2cat is not None and idx2cat[i]:  # check categorical feature
                    colored_feature = False
                else:
                    values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except Exception:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax:  # fixes rare numerical precision issues
                    vmin = vmax

                # vmin = 0.0
                # vmax = 0.7

                assert values.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values) | (values < 0.05)
                # pl.scatter(
                #     shaps[nan_mask],
                #     pos + ys[nan_mask],
                #     color="#777777",
                #     s=16,
                #     alpha=alpha,
                #     linewidth=0,
                #     zorder=3,
                #     rasterized=len(shaps) > 500,
                # )

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                # print(cvals)
                pl.scatter(
                    shaps[np.invert(nan_mask)],
                    pos + ys[np.invert(nan_mask)],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=16,
                    c=cvals,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    # rasterized=len(shaps) > 500,
                )
            else:
                pl.scatter(
                    shaps,
                    pos + ys,
                    s=16,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    color=color if colored_feature else "#777777",
                    rasterized=len(shaps) > 500,
                )
    else:
        raise ValueError(plot_type)

    # draw the color bar
    if (
            color_bar
            and features is not None
            and plot_type != "bar"
            and (plot_type != "layered_violin" or color in pl.cm.datad)
    ):
        import matplotlib.cm as cm

        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.ax.set_yticks([0, 1], labels=['0', '100'])
        cb.outline.set_visible(False)
    #         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    #         cb.ax.set_aspect((bbox.height - 0.9) * 20)
    # cb.draw_all()

    pl.gca().xaxis.set_ticks_position("bottom")
    pl.gca().yaxis.set_ticks_position("none")
    pl.gca().spines["right"].set_visible(False)
    pl.gca().spines["top"].set_visible(False)
    pl.gca().spines["left"].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    if plot_type != "bar":
        pl.gca().tick_params("y", length=20, width=0.5, which="major")
    pl.gca().tick_params("x", labelsize=11)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)
    else:
        pl.xlabel(labels["VALUE"], fontsize=13)
    pl.tight_layout()
    if show:
        pl.show()


def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text


def is_color_map(color):
    safe_isinstance(color, "matplotlib.colors.Colormap")
