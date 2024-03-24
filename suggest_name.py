import argparse
import pickle
import re
import subprocess
from pathlib import Path
from timeit import timeit
from typing import NamedTuple, Any

import epitran
import epitran.vector
import pandas
from tqdm import tqdm
import panphon.distance
import networkx
import structlog
import jinja2
import matplotlib.pyplot as plt

logger = structlog.stdlib.get_logger()


class CharacterInfo(NamedTuple):
    character_category: str
    is_upper: int
    orthographic_form: str
    phonetic_form: str
    segments: list[tuple]


def load_names() -> pandas.DataFrame:
    logger.info("Loading names")
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

    logger.info("Names loaded")
    return all_names


def calculate_distances(valid_names: pandas.DataFrame) -> pandas.DataFrame:
    logger.info("Calculating distances")
    if Path('distances.csv').is_file():
        logger.warning("Distances already calculated, delete the file if you want to recalculate")
        return pandas.read_csv('distances.csv', index_col=0)

    distance = panphon.distance.Distance()
    distances = pandas.DataFrame()
    distances.index = valid_names.loc[:, 'ipa']
    for index1 in tqdm(distances.index, desc="Calculating distances", unit="names"):
        distances.loc[:, index1] = valid_names.set_index('ipa', drop=False).loc[:, 'ipa'].apply(
            lambda index2: distance.weighted_feature_edit_distance(index1, index2)
        )
    distances.to_csv('distances.csv')
    logger.info("Distances calculated")
    return distances


def percentile(name: str, distances: pandas.DataFrame, percent: float) -> pandas.Series:
    select_column: pandas.Series = distances.loc[:, name]
    limit = select_column.quantile(percent)
    return select_column[select_column <= limit]


def create_graph(distances: pandas.DataFrame, percent: float = .0005):  # 100.05
    logger.info("Creating graph")
    graph = networkx.Graph()
    for column in tqdm(distances.columns, desc="Creating graph", unit="names"):
        selected = percentile(column, distances, percent)
        for index, value in selected.items():
            if value == 0:
                continue
            graph.add_edge(column, index, weight=value)
    logger.info("Graph created")
    logger.info("Graph has %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
    logger.info("Graph average degree of nodes is %f", sum(dict(graph.degree).values()) / graph.number_of_nodes())
    return graph


def detect_communities(distances: pandas.DataFrame) -> list:
    logger.info("Detecting communities")
    if Path('communities.pkl').is_file():
        logger.warning("Communities already detected, delete the file if you want to recalculate")
        with open('communities.pkl', 'rb') as file:
            return pickle.load(file)

    graph = create_graph(distances)
    communities = networkx.algorithms.community.louvain_communities(graph)
    with open('communities.pkl', 'wb') as file:
        pickle.dump(communities, file)
    logger.info("Communities detected")
    logger.info("There are %d communities", len(communities))
    logger.info("The average community size is %f", sum(len(community) for community in communities) / len(communities))
    return communities


def community_id_of_name(name: str, communities: list[set[str]]) -> int:
    for index, community in enumerate(communities):
        if name in community:
            return index
    raise ValueError(f"Name {name} not found in any community")


def similarity_to(name_ipa1: str, name_ipa2: str, distances: pandas.DataFrame) -> float:
    return distances.loc[name_ipa1, name_ipa2]


def name_entry_from_name(
    name: str, reference_name: str,
    valid_names: pandas.DataFrame, distances: pandas.DataFrame
) -> dict[str, Any]:
    reference_entry = valid_names.loc[valid_names.loc[:, 'name'] == reference_name].iloc[0].to_dict()
    entry = valid_names.loc[valid_names.loc[:, 'name'] == name].iloc[0].to_dict()
    entry['similarity'] = similarity_to(entry['ipa'], reference_entry['ipa'], distances)
    return entry


def name_entries_from_names(
    names: list[str], reference_name: str,
    valid_names: pandas.DataFrame, distances: pandas.DataFrame
) -> list[dict[str, Any]]:
    return [name_entry_from_name(name, reference_name, valid_names, distances) for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', type=str, nargs='+', help='Reference name')
    args = parser.parse_args()

    valid_names = load_names()
    distances = calculate_distances(valid_names)
    communities = detect_communities(distances)
    valid_names.loc[:, 'community'] = valid_names.loc[:, 'ipa'].apply(
        lambda name: community_id_of_name(name, communities)
    )
    similar_count = 30

    reference_names = args.names

    similar_names = {}
    names_of_reference_community = {}
    for name in reference_names:
        entry = name_entry_from_name(name, name, valid_names, distances)

        ax: plt.Axes
        fig, ax = plt.subplots()
        ax.hist(distances.loc[:, entry['ipa']], bins=42)
        ax.set_title(f'Distribution of distances to {name}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Count')
        plt.savefig(f'{name}_histogram.png')
        plt.close(fig)

        ax: plt.Axes
        fig, ax = plt.subplots()
        ax.plot(distances.loc[:, entry['ipa']].sort_values().values)
        ax.set_title(f'Distances to {name}')
        ax.set_xlabel('Name')
        ax.set_ylabel('Distance')
        plt.savefig(f'{name}_distances.png')
        plt.close(fig)

        similar_ipas = distances.loc[:, entry['ipa']].sort_values().head(similar_count).index.tolist()
        similar_names[name] = list(
            valid_names.loc[valid_names.loc[:, 'ipa'] == ipa, 'name'].iloc[0] for ipa in similar_ipas
        )
        reference_community = valid_names.loc[valid_names.loc[:, 'name'] == name, 'community'].values[0]
        names_of_reference_community[name] = valid_names.loc[
            valid_names.loc[:, 'community'] == reference_community,
            'name'
        ].tolist()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader('template'),
        undefined=jinja2.StrictUndefined,
    )
    env.globals['valid_names'] = valid_names
    env.globals['distances'] = distances
    env.globals['similarity_to'] = similarity_to
    env.globals['name_entry_from_name'] = name_entry_from_name
    env.globals['name_entries_from_names'] = name_entries_from_names
    template = env.get_template('report.html')
    with open('report.html', 'w', encoding='utf-8') as file:
        file.write(
            template.render(
                similar_names=similar_names,
                names_of_reference_community=names_of_reference_community,
            )
        )

    print()


if __name__ == '__main__':
    main()
