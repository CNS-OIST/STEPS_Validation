import logging
import os
import shutil

import numpy
import pandas


class TraceDBError(Exception):
    pass


class TraceDB:
    """Trace DataBase

    This class maintains all the traces of a particular simulation type (sample or benchmark)
    """

    def __init__(
        self,
        name,
        traces,
        folder_path="",
        time_trace="t",
        clear_raw_traces_cache=False,
        clear_refined_traces_cache=False,
        is_refine=True,
        save_raw_traces_cache=True,
        save_refined_traces_cache=True,
    ):
        """Init

        Args:
              - traces: traces that must be fed with the raw data found in folder_path
              - folder_path (str): path to where the raw data are. The raw data must be in the form of
              a matrix where each column is a trace (in the order specified in traces). More information in
              _extract_raw_data
              - time_trace specified a particular trace as the time trace
        """
        self.name = name
        self.time_trace = time_trace
        self.root_traces = [i.name for i in traces if not i.derivation_params]
        self.save_raw_traces_cache = save_raw_traces_cache
        self.save_refined_traces_cache = save_refined_traces_cache

        if isinstance(traces, list):
            self.traces = {i.name: i for i in traces}
        elif isinstance(traces, dict):
            self.traces = traces
        else:
            raise TraceDBError(f"{traces} is neither a list nor a dict")

        if len(traces) != len(self.traces):
            raise TraceDBError(
                "Some traces have the same name. Please make them unique"
            )

        self.folder_path = folder_path

        self._extract_all_raw_data(clear_raw_traces_cache)

        self._check_for_nans()

        if is_refine:
            self._refine(clear_refined_traces_cache)

        for i in self.traces:
            self.traces[i]._update_raw_traces_common_prefix_and_suffix()

    def get_time_trace(self):
        return self.traces.get(self.time_trace, None)

    def __str__(self, interesting_traces=[], is_full_display=False):
        if is_full_display:
            pandas.set_option("display.max_rows", None, "display.max_columns", None)

        ss = f"folder_path: {self.folder_path}\n"
        if not interesting_traces:
            interesting_traces = self.traces.keys()
        elif isinstance(interesting_traces, str):
            interesting_traces = [interesting_traces]
        for k in interesting_traces:
            ss += str(self.traces[k]) + "\n"
        return ss

    def __len__(self):
        return len(self.traces)

    def rewrite_raw_traces(self, suffix="_1"):
        """Rewrite raw trace files after they were loaded

        Useful to convert dimensions
        """

        if len(suffix) == 0:
            cont = input(
                "I am going to re-write the raw traces. Are you sure you want to proceed? [y/n] (This "
                "operation cannot "
                "be reversed)"
            )

            if cont != "y":
                logging.warning("Aborted")
                return

            logging.warning("I am going to rewrite the raw traces")

        for file_path in next(iter(self.traces.values())).raw_traces.keys():
            df = pandas.DataFrame(
                {k: v.raw_traces[file_path] for k, v in self.traces.items()}
            )
            base_file_path, ext = os.path.splitext(file_path)
            final_path = f"{base_file_path}{suffix}{ext}"
            logging.warning(f"rewriting {final_path} ...")
            df.to_csv(final_path, sep=" ", index=False)

    def _derive_raw_data(self):
        """Service function to call derive on all traces"""
        for trace_name in self.traces:
            self.traces[trace_name].derive_raw_data(self.traces)

    def _check_for_nans(self):
        """Check for nans in the raw traces"""
        for i in self.traces.values():
            i._check_for_nans()

    def _refine(self, clear_cache=False):
        """Service function that calls refine on all the traces"""

        cache_path = os.path.join(self.folder_path, "refined_traces_cache/")

        if clear_cache:
            try:
                shutil.rmtree(cache_path)
            except OSError:
                pass

        if self.save_refined_traces_cache:
            os.makedirs(cache_path, exist_ok=True)

        for trace_name in self.traces:
            try:
                self.traces[trace_name].refined_traces_from_parquet(cache_path)
            except:
                self.traces[trace_name].refine(self.get_time_trace())
                if self.save_refined_traces_cache:
                    self.traces[trace_name].refined_traces_to_parquet(cache_path)

    def _extract_all_raw_data(self, clear_cache=False):
        """Extract all raw data from the files in self.folder_path (not recursive)"""

        if not self.folder_path:
            return

        cache_path = os.path.join(self.folder_path, "raw_traces_cache/")

        if clear_cache:
            try:
                shutil.rmtree(cache_path)
            except OSError:
                pass

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        try:
            for k in self.traces.keys():
                self.traces[k].raw_traces_from_parquet(cache_path)
            logging.info("Successfully loaded from cache")
        except:
            root, _, files = next(os.walk(self.folder_path))
            files.sort()

            print(self.folder_path)

            for f in files:
                file_path = os.path.join(root, f)
                self._extract_raw_data(file_path)

            self._derive_raw_data()

        if self.save_raw_traces_cache:
            for v in self.traces.values():
                v.raw_traces_to_parquet(cache_path)

    def _extract_raw_data(self, file_path: str):
        """Extract raw data from a file

        This function reads the file and tries to extract the raw data supposing we have a matrix where each column
        is a trace ordered as self.traces (without the derived traces).

        Note: we assume that every row that presents that number of columns with
        floats is valid.

        Args:
            file_path (str): file from which data must be extracted
        """

        logging.info(f"Extract {file_path}")

        header = -1
        with open(file_path) as f:
            for ls in f.readlines():
                try:
                    spl = [float(x) for x in ls.split()]

                    if len(spl) < len(self.root_traces):
                        raise ValueError

                    if len(spl) > len(self.root_traces):
                        raise TraceDBError(
                            f"The file {file_path} contains more traces ({len(spl)}) than the root "
                            f"traces "
                            f"provided ({len(self.root_traces)}): {', '.join(self.root_traces)}"
                        )
                    break
                except ValueError:
                    header += 1

        data = pandas.read_csv(
            file_path,
            delim_whitespace=True,
            names=self.root_traces,
            header=None,
            skiprows=range(header + 1),
        )

        data = data.apply(pandas.to_numeric, errors="coerce").dropna()

        for k, v in data.items():
            self.traces[k].add_raw_trace(file_path, v)

    def raw_rows(self):
        """Maximum number of rows per raw trace"""
        return numpy.array([len(trace.raw_traces) for trace in self.traces.values()])

    def plot(
        self, trace_names=[], trace_files=[], savefig_path=None, legend=False, suffix=""
    ):
        """Plot

        Args:
              - trace_names (list, optional): if specified it plots all the raw_traces of the traces specified.
              Otherwise we plot everything
              - savefig_path (str, optional): if speficied we also save the file in the specified folder.
        """

        time_trace = self.get_time_trace()

        if not trace_names:
            trace_names = self.traces

        for trace_name in trace_names:
            trace = self.traces[trace_name]
            if trace.name == time_trace.name:
                continue

            trace.plot(
                trace_files=trace_files,
                time_trace=time_trace,
                savefig_path=savefig_path,
                title_prefix=self.name,
                legend=legend,
                suffix=suffix,
            )
