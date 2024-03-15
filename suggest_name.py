import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import epitran
import epitran.vector
import pandas
from tqdm import tqdm


class CharacterInfo(NamedTuple):
    character_category: str
    is_upper: int
    orthographic_form: str
    phonetic_form: str
    segments: list[tuple]


def load_names() -> pandas.DataFrame:
    male_names = pandas.read_csv('male.txt', header=None, names=['name'])
    male_names.loc[:, 'sex'] = 'male'
    female_names = pandas.read_csv('female.txt', header=None, names=['name'])
    female_names.loc[:, 'sex'] = 'female'
    all_names = pandas.concat((male_names, female_names), axis=0)
    all_names.reset_index(drop=True, inplace=True)
    tqdm.pandas(desc="Transcribing names", unit="names")

    epi = epitran.Epitran('hun-Latn')
    all_names.loc[:, 'ipa'] = all_names.loc[:, 'name'].progress_apply(
        lambda name: epi.transliterate(name)
    )
    all_names.loc[:, 'word_tuples'] = all_names.loc[:, 'name'].progress_apply(
        lambda name: tuple(CharacterInfo(*raw) for raw in epi.word_to_tuples(name))
    )
    all_names.loc[:, 'segments'] = all_names.loc[:, 'word_tuples'].progress_apply(
        lambda infos: tuple(segment for info in infos for segment in info.segments)
    )

    return all_names


def main():
    valid_names = load_names()
    print()


if __name__ == '__main__':
    main()
