import logging
import os
import shutil

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
        clear_refined_traces_cache=False,
        save_refined_traces_cache=True,
        keep_raw_traces=False,
        ban_list=[],
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
        self.clear_refined_traces_cache = clear_refined_traces_cache
        self.save_refined_traces_cache = save_refined_traces_cache
        self.keep_raw_traces = keep_raw_traces
        self.ban_list = set(ban_list)

        if isinstance(traces, list):
            self.traces = {i.name: i for i in traces}
        elif isinstance(traces, dict):
            self.traces = traces
        else:
            raise TraceDBError(f"{traces} is neither a list nor a dict")

        if len(traces) != len(self.traces):
            raise TraceDBError(
                "Some traces have the same name. Please, make them unique"
            )

        self.folder_path = folder_path
        self.refined_traces_cache_path = os.path.join(
            folder_path, "refined_traces_cache"
        )

        if self._load_refined_traces():
            self._extract_and_refine()

        if self.save_refined_traces_cache:
            self._save_refined_traces_cache()

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

        if not self.keep_raw_traces:
            if not self.keep_raw_traces:
                raise TraceDBError(
                    f"Rewriting raw traces is possible only if `keep_raw_traces` is marked as True in TraceDB."
                )

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

    def _check_for_nans(self):
        """Check for nans in the raw traces"""
        for i in self.traces.values():
            i._check_for_nans()

    def _save_refined_traces_cache(self):
        """ Save refined traces cache """
        os.makedirs(self.refined_traces_cache_path, exist_ok=True)

        for trace_name in self.traces:
            self.traces[trace_name].refined_traces_to_parquet(
                self.refined_traces_cache_path
            )

    def _derive_raw_data(self):
        if not self.keep_raw_traces:
            if not self.keep_raw_traces:
                raise TraceDBError(
                    f"Deriving raw traces is possible only if `keep_raw_traces` is marked as True in TraceDB."
                )

        """Service function to call derive on all traces"""
        for trace_name in self.traces:
            self.traces[trace_name].derive_raw_data(self.traces)

    def _load_refined_traces(self):
        cache_path = os.path.join(self.folder_path, "refined_traces_cache/")
        if self.clear_refined_traces_cache:
            try:
                shutil.rmtree(cache_path)
            except OSError:
                pass

        extraction_needed = False
        for trace_name in self.traces:
            try:
                self.traces[trace_name].refined_traces_from_parquet(cache_path)
            except:  # catch all the possible errors from refined_traces_from_parquet
                extraction_needed = True

        return extraction_needed

    def _extract_and_refine(self):

        if not self.folder_path:
            return

        logging.info(f"Extract and refine from folder: {self.folder_path}")
        root, _, files = next(os.walk(self.folder_path))
        files.sort()

        for f in files:
            name, ext = os.path.splitext(f)
            if ext != ".txt" or name in self.ban_list:
                continue

            file_path = os.path.join(root, f)
            self._extract_raw_data(file_path)

            self._check_for_nans()

            for trace in self.traces.values():
                trace.refine(file_path, self.get_time_trace())

            if not self.keep_raw_traces:
                for trace in self.traces.values():
                    trace.remove_raw_traces()

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

    def plot(
        self,
        trace_names=[],
        trace_files=[],
        savefig_path=None,
        legend=False,
        suffix="",
        *argv,
        **kwargs,
    ):
        """Plot

        Args:
              - trace_names (list, optional): if specified it plots all the raw_traces of the traces specified.
              Otherwise we plot everything
              - savefig_path (str, optional): if speficied we also save the file in the specified folder.
        """

        if not self.keep_raw_traces:
            raise TraceDBError(
                f"Plotting raw traces is possible only if `keep_raw_traces` is marked as True in TraceDB."
            )

        time_trace = self.get_time_trace()

        if not trace_names:
            trace_names = list(self.traces.keys())
            print(trace_names)

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
                *argv,
                **kwargs,
            )
