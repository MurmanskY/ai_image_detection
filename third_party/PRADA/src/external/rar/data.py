"""Stub of upstream data module (not vendored in PRADA).

These dataset classes are only needed for training; inference via
`create_pretrained_tokenizer` never instantiates them. They raise if used.
"""

class _UnavailableDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "external.rar.data datasets are not vendored in PRADA; "
            "they are only required for training."
        )


class SimpleImageDataset(_UnavailableDataset):
    pass


class PretoeknizedDataSetJSONL(_UnavailableDataset):
    pass


class PretokenizedWebDataset(_UnavailableDataset):
    pass
