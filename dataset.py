"""
@author: tsdj

"""


from functools import partial

import torch

from torch import Tensor
from torch.utils.data import Dataset

import pandas as pd


CHARS_IN_TOYDATA = [
    ' ',
    '"',
    "'",
    '(',
    ')',
    '*',
    '+',
    ',',
    '-',
    '.',
    '/',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    ':',
    ';',
    '?',
    '@',
    '[',
    ']',
    '_',
    '`',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    '{',
    '¢',
    '£',
    '©',
    '¬',
    'Â',
    'Ã',
    'â',
    'œ',
    'š',
    'ž',
    '‚',
    '„',
    '€',
]
MAP_CHAR_IDX = {char: idx for idx, char in enumerate(CHARS_IN_TOYDATA, start=2)}

def simple_encoder(hisco: str, max_len: int) -> list[int]:
    encoded = [MAP_CHAR_IDX.get(char, 0) for char in hisco]
    encoded = encoded[:max_len]
    encoded += [1] * (max_len - len(encoded))

    return encoded


class HISCODataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__()

        self.dataset = dataset
        self.tokenizer = partial(simple_encoder, max_len=32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, str | Tensor]:
        record = self.dataset.iloc[item]
        encoded = self.tokenizer(record.occ1)

        package = {
            'occ1': record.occ1,
            'encoded': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(record.label, dtype=torch.long),
        }

        return package
