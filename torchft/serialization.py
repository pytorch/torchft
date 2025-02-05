import pickle
from dataclasses import dataclass
from io import BufferedIOBase
from typing import Dict, Tuple

import torch


@dataclass
class _Entry:
    key: str
    dtype: object
    is_storage: bool
    length: int


class _InMemoryStateDict:
    def __init__(self) -> None:
        self.records: Dict[str, Tuple[object, int]] = {}

    def write_record(self, key: str, data: object, length: int) -> None:
        self.records[key] = (data, length)

    def write_to(self, f: BufferedIOBase) -> None:
        entries = []
        for key, (data, length) in self.records.items():
            entries.append(
                _Entry(
                    key=key,
                    is_storage=isinstance(data, torch.UntypedStorage),
                    dtype=type(data),
                    length=length,
                )
            )

        pickle.dump(entries, f)

        for key, (data, length) in self.records.items():
            if isinstance(data, bytes):
                f.write(data)
            elif isinstance(data, str):
                f.write(data.encode("utf-8"))
            elif isinstance(data, torch.UntypedStorage):
                data._write_file(f, False, False, 1)
            else:
                raise TypeError(f"unknown type: {type(data)}")

    def read_from(self, f: BufferedIOBase) -> None:
        entries = pickle.load(f)

        for entry in entries:
            data = f.read(entry.length)
            if entry.is_storage:
                storage = torch.frombuffer(
                    data,
                    dtype=torch.uint8,
                ).untyped_storage()

                self.records[entry.key] = (
                    storage,
                    entry.length,
                )
            else:
                self.records[entry.key] = (data, entry.length)

    def has_record(self, key: str) -> bool:
        return key in self.records

    def get_record(self, key: str) -> object:
        return self.records[key][0]

    def get_storage_from_record(
        self, key: str, _length: int, _type: int
    ) -> torch.Tensor:
        return torch.tensor(self.records[key][0], dtype=torch.uint8)

    def serialization_id(self) -> str:
        return "torchft"


def streaming_save(obj: object, f: BufferedIOBase) -> None:
    out = _InMemoryStateDict()
    torch.serialization._save(
        obj,
        zip_file=out,
        pickle_module=pickle,
        pickle_protocol=2,
        _disable_byteorder_record=False,
    )
    out.write_to(f)


def streaming_load(f: BufferedIOBase) -> object:
    out = _InMemoryStateDict()
    out.read_from(f)
    return torch.serialization._load(
        zip_file=out,
        map_location=None,
        pickle_module=pickle,
    )
