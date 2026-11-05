"""Type the ONNX Runtime 1.31.1 CPU inference API used by the speaker encoder."""

import numpy as np
from numpy.typing import NDArray

class SessionOptions:
    inter_op_num_threads: int
    intra_op_num_threads: int
    def __init__(self) -> None: ...

class InferenceSession:
    def __init__(
        self, path_or_bytes: str, *, sess_options: SessionOptions, providers: list[str]
    ) -> None: ...
    def run(
        self, output_names: list[str], input_feed: dict[str, NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]: ...
