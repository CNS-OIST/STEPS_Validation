import os
import re

import matplotlib.pyplot as plt


class Figure:
    """ Generic figure wrapper to automatically set parameters for all the subplots """

    def __init__(
        self,
        title=None,
        suffix=None,
        savefig_path=None,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        fontsize=None,
        legendfontsize=None,
        interactive=True,
        pplot=None,
    ):
        self.title = title
        self.suffix = suffix
        self.savefig_path = savefig_path
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fontsize = fontsize
        self.legendfontsize = legendfontsize
        self.interactive = interactive
        self.pplot = pplot

        if self.pplot is None:
            self.fig, self.pplot = plt.subplots(1, 1)

    def finalize(self, with_legend=True):
        """Try to save and show if possible"""

        self.pplot.set_xlim(self.xlim)
        self.pplot.set_ylim(self.ylim)

        if with_legend:
            self.set_legend()

        try:
            self.fig.tight_layout()
            self.savefig(self.savefig_path, self.title, self.fig, self.suffix)
            if self.interactive:
                self.fig.show()
        except AttributeError:
            pass

    def set_legend(self):
        """Try to set the legend either to a figure or to a figure with subfigures"""
        try:
            plt.setp(self.pplot.get_legend().get_texts(), fontsize=self.legendfontsize)
        except AttributeError:
            self.pplot.legend(fontsize=self.legendfontsize)

    def set_xlabel(self, default):
        """Set xlabel"""
        if self.xlabel is None:
            self.xlabel = default

        if len(self.xlabel):
            self.pplot.set_xlabel(self.xlabel, fontsize=self.fontsize)
        return self.xlabel

    def set_ylabel(self, default):
        """Set ylabel"""
        if self.ylabel is None:
            self.ylabel = default

        if len(self.ylabel):
            self.pplot.set_ylabel(self.ylabel, fontsize=self.fontsize)
        return self.ylabel

    def set_xticks(self, xx, labels):
        """Set xticks"""
        self.pplot.set_xticks(xx)
        self.pplot.set_xticklabels(labels)
        self.pplot.tick_params(labelsize=self.fontsize)

    def set_title(self, default):
        """Set title"""
        if self.title is None:
            self.title = default

        if len(self.title):
            try:
                self.pplot.title(self.title, fontsize=self.fontsize)
            except TypeError:
                self.pplot.set_title(self.title, fontsize=self.fontsize)
        return self.title

    @staticmethod
    def savefig(savefig_path, file_name, fig, suffix=""):
        """Routine to decide where and how save the figure"""
        if savefig_path is None:
            return

        if suffix:
            file_name += "_" + suffix
        file_name = re.sub(" ", "_", file_name)
        file_name = re.sub("\.", "", file_name)
        file_name += ".jpg"
        os.makedirs(savefig_path, exist_ok=True)

        fig.savefig(os.path.join(savefig_path, file_name), dpi=300)
