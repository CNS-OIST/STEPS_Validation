import os


class RawDataError(Exception):
    pass


class RawData:
    """Data RawData

    collection of all the methods used to extract data raw in a standardized dict
    """

    def __init__(self, arg0, traces: list = []):
        """Init

        Args:
            folder (str): in which the various raw data are
            traces (list): data ordering. Check _extract_raw_data for more info
        """
        self.traces = traces
        self.folder = ""
        self.data = {}
        if isinstance(arg0, str):
            if not traces:
                raise RawDataError("traces is empty")
            self.traces = traces
            self.folder = arg0
            self.data = self._extract_all_raw_data()
        elif isinstance(arg0, dict):
            self.data = arg0
        else:
            raise RawDataError(f"unknown arg0 of type {type(arg0).__name__}")

    def __str__(self):
        ss = f"traces: {str(self.traces)}"
        ss += f"folder: {self.folder}"
        ss += "data:"
        ss += str(self.data)
        return ss

    def _extract_raw_data(self, file: str):
        """Data RawData

        This function reads the file and tries to extract the raw data supposing that
        traces are in a table like shape space separated. For example, we indicate to
        the code that we want to extract the traces = ['t', 'x', 'y']. The file should
        present them column-wise:

        0 1.0 2.0
        0.1 1.0 2.0
        0.2 1.0 2.0
        [...]

        where the first column represents 't', the second 'x' and the third 'y'

        Note: we assume that every row that presents that number of columns with
        floats is valid.

        Args:
            file (str): file from which data must be extracted
            traces (list): data ordering

        Returns:
            dict: traces dictionary -> "trace_name" : [values]
        """

        out = {i: [] for i in self.traces}

        f = open(file)
        for l in f.readlines():
            try:
                ls = l.split()
                if len(ls) == len(self.traces):
                    for idx, trace in enumerate(self.traces):
                        out[trace].append(float(ls[idx]))
            except:
                pass

        if len(next(iter(out.values()))) == 0:
            raise RawDataError(
                "No trace was extracted. Probably the file structure does not match the `traces` list"
            )

        return out

    def _extract_all_raw_data(self):
        """Extract all raw data from the files in a folder (not recursive)

        Returns:
            dict of dicts: "file_path" : "trace_name" : [values]
        """

        out = {}

        for root, _, files in os.walk(self.folder):
            if root == self.folder:
                for f in files:
                    path = os.path.join(root, f)
                    out[path] = self._extract_raw_data(path)

        return out
