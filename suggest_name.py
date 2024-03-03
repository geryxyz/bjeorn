import re
import subprocess
from pathlib import Path

import pandas
from tqdm import tqdm


def phonetic_transcribe(name: str) -> tuple[str, ...]:
    name = name.lower().replace('ő', 'ö').replace('ű', 'ü')  # espeak does not support hungarian special characters :(
    raw_transcript = subprocess.check_output(f"espeak -q -X -v hu '{name}'", shell=True, encoding="utf-8")
    transcript = []
    for line in raw_transcript.split('\n'):
        line = line.strip()
        if "\t" in line:
            columns = re.split(r'\s+', line)
            transcript.append(columns[2].removeprefix('[').removesuffix(']'))
    return tuple(transcript)


def load_names() -> pandas.DataFrame:
    if Path('names.csv').is_file():
        return pandas.read_csv('names.csv')

    male_names = pandas.read_csv('male.txt', header=None, names=['name'])
    male_names.loc[:, 'sex'] = 'male'
    female_names = pandas.read_csv('female.txt', header=None, names=['name'])
    female_names.loc[:, 'sex'] = 'female'
    all_names = pandas.concat((male_names, female_names), axis=0)
    all_names.reset_index(drop=True, inplace=True)
    tqdm.pandas(desc="Transcribing names", unit="names")
    all_names.loc[:, 'transcript'] = all_names.loc[:, 'name'].progress_apply(phonetic_transcribe)
    all_names.to_csv('names.csv', index=False)

    return all_names


def main():
    valid_names = load_names()


if __name__ == '__main__':
    main()