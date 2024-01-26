from __future__ import annotations

import re
from collections import UserDict
from enum import Enum, StrEnum
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import click
import lkml
from graphviz import Digraph, Graph

INPUT_MODELS_PATH = "../discord-looker-revenue"
MODEL_FILE_EXTENSION = "*.model.lkml"
VIEW_FILE_EXTENSION = "*.view.lkml"
OUTPUT_FILE_PATH = "output/dependency_graph.gv"
RENDER_FORMAT = "png"


class Node(UserDict):
    def __init__(self):
        self.data = dict(depends_on=[])


class NodeType(StrEnum):
    MODEL = "model"
    EXPLORE = "explore"
    VIEW = "view"
    TABLE = "table"


class DepNode:
    def __init__(self, node_type: NodeType, name: str, filepath: Optional[Path] = None):
        self.type = node_type
        self.dependencies: set[DepNode] = set()
        self.name = name
        self.filepath = filepath

    def depends_on(self, node: DepNode):
        self.dependencies.add(node)

    def directory_label(self, root: Path) -> str:
        return str(self.filepath.parent.resolve()).replace(str(root), '').strip('/') + '/'

    @property
    def path_key(self) -> str:
        return DepNode.path_key_for_node(self.type, self.name)

    @staticmethod
    def path_key_for_node(node_type: NodeType, name: str) -> str:
        return f"{node_type.value}.{name}"


class Nodes:
    def __init__(self):
        self.nodes: dict[str, DepNode] = dict()

    def create_node(self, node_type: NodeType, name: str, filepath: Optional[Path] = None) -> DepNode:
        node = DepNode(node_type, name, filepath)
        self.nodes[node.path_key] = node
        return node

    def find_or_create_node(self, node_type: NodeType, name: str, filepath: Optional[Path] = None) -> Optional[DepNode]:
        node = self.nodes.get(DepNode.path_key_for_node(node_type, name)) or self.create_node(node_type, name, filepath)

        if node and filepath:
            node.filepath = filepath

        return node


def read_lookml(path: Path) -> dict:
    """Parse a LookML file"""
    with open(path) as f:
        lookml = lkml.load(f.read())
    return lookml


def build_model_nodes(nodes: Nodes, model: Path) -> Nodes:
    """
    Build nodes dictionary from a LookML model.
    Each node has a unique name and a list of node names it depends on.

    Returns:
        "nodes": {
            "model.product": {
                "depends_on": [
                        "explore.user_events_cube"
                ]
            },
            ...
        }
    """
    lookml = read_lookml(model)
    model_name = model.name.split(".")[0]

    model_node = nodes.find_or_create_node(NodeType.MODEL, model_name, model)

    for explore in lookml["explores"]:
        explore_node = nodes.find_or_create_node(NodeType.EXPLORE, explore['name'], model)

        if "joins" in explore.keys():
            for view in explore["joins"]:
                explore_node.depends_on(nodes.find_or_create_node(NodeType.VIEW, view.get('from') or view['name'], model))

        explore_node.depends_on(nodes.find_or_create_node(
            NodeType.VIEW,
            explore.get('from') or explore.get('view_name') or explore[
                'name'],
            model
        ))

        model_node.depends_on(explore_node)

    return nodes


TABLE_REFERENCE_REGEX = re.compile(r'\$\{\s*([a-zA-Z0-9_]+)\.SQL_TABLE_NAME\s*}')
IMPERFECT_TABLE_REGEX = re.compile(r'(JOIN|FROM)\s*(`[`a-zA-Z0-9_.-]+)\s*', re.IGNORECASE | re.MULTILINE)


def build_view_nodes(nodes: Nodes, view_path: Path, tables: bool) -> Nodes:
    """
    Build nodes dictionary from a LookML model.
    Each node has a unique name and a list of node names it depends on.

    Returns:
        "nodes": {
            "model.product": {
                "depends_on": [
                        "explore.user_events_cube"
                ]
            },
            ...
        }
    """
    lookml = read_lookml(view_path)

    for view in lookml['views']:
        # TODO: support non sql defined derived tables

        view_node = nodes.find_or_create_node(NodeType.VIEW, view['name'], view_path)

        if 'sql_table_name' in view and tables:
            view_node.depends_on(nodes.find_or_create_node(NodeType.TABLE, view['sql_table_name'], view_path))

        if 'derived_table' in view:
            assert 'sql' in view['derived_table']

            sql = view['derived_table']['sql']

            # find views
            for match in TABLE_REFERENCE_REGEX.finditer(sql):
                view_name = match.groups()[0]

                view_node.depends_on(nodes.find_or_create_node(NodeType.VIEW, view_name, view_path))

            if tables:
                for match in IMPERFECT_TABLE_REGEX.finditer(sql):
                    table_name = match.groups()[1].replace('`', '')

                    view_node.depends_on(nodes.find_or_create_node(NodeType.TABLE, table_name, view_path))

        if 'extends__all' in view:
            for extends in view['extends__all']:
                for extended_view in extends:
                    view_node.depends_on(nodes.find_or_create_node(NodeType.VIEW, extended_view))

    return nodes


def build_child_map(nodes: Nodes) -> dict:
    """
    Build child map combining all `depends_on` specifications.
    Each entry is a node with the full list of children.

    Returns:
        "child_map": {
            "explore.user_events_cube": [
                "view.user_events_cube",
                "view.dummy_view"
            ],
            ...
        }
    """
    child_map = dict()
    for path_key, contents in nodes.nodes.items():
        child_map[path_key] = [n.path_key for n in contents.dependencies]
    return child_map


def build_manifest(nodes: Nodes, filters: Optional[list] = None, tables: bool = True) -> dict:
    """
    Build manifest containing nodes and child map objects.
    Expects LookML .model files to be in input/models folder.
    """
    p = Path(INPUT_MODELS_PATH)
    models = list(p.glob(f"**/{MODEL_FILE_EXTENSION}"))

    if not models:
        raise FileNotFoundError(
            f"No {MODEL_FILE_EXTENSION} files found in {INPUT_MODELS_PATH}"
        )

    views = list(p.glob(f"**/{VIEW_FILE_EXTENSION}"))

    if not views:
        raise FileNotFoundError(
            f"No {VIEW_FILE_EXTENSION} files found in {INPUT_MODELS_PATH}"
        )

    for model in models:
        build_model_nodes(nodes, model)

    for view in views:
        build_view_nodes(nodes, view, tables)

    child_map = build_child_map(nodes)

    if filters:
        filtered_child_map = {}

        made_change = True
        parents = {f for f in filters}

        while made_change:
            made_change = False

            for path_key, children in child_map.items():
                if path_key in parents and path_key not in filtered_child_map:
                    filtered_child_map[path_key] = children
                    [parents.add(c) for c in children]
                    made_change = True
                    continue

        child_map = filtered_child_map

    return {"child_map": child_map}


def build_graph(nodes: Nodes, manifest: dict, no_tables: bool) -> Digraph:
    """
    Build directed graph of dependencies.
    Add edges by iterating over each parent/child combination in manifest child map.
    """

    prefix_path = Path(INPUT_MODELS_PATH).resolve()

    model_nodes = [n for n in nodes.nodes.values() if n.type == NodeType.MODEL]
    explore_nodes = [n for n in nodes.nodes.values() if n.type == NodeType.EXPLORE]
    view_nodes = [n for n in nodes.nodes.values() if n.type == NodeType.VIEW]
    table_nodes = [n for n in nodes.nodes.values() if n.type == NodeType.TABLE]

    table_nodes_by_project_by_dataset: dict[str, dict[str, list[DepNode]]] = {}

    if not no_tables:
        for node in table_nodes:
            project, dataset, table = node.name.split('.')

            if project not in table_nodes_by_project_by_dataset:
                table_nodes_by_project_by_dataset[project] = {}

            if dataset not in table_nodes_by_project_by_dataset[project]:
                table_nodes_by_project_by_dataset[project][dataset] = []

            table_nodes_by_project_by_dataset[project][dataset].append(node)

    model_cluster = Digraph("cluster_models",
                            graph_attr={"style": "solid", "rank": "same", "label": "Models"},
                            node_attr={"shape": "box"})
    explore_cluster = Digraph("cluster_explores",
                              graph_attr={"style": "solid", "rank": "same", "label": "Explores"},
                              node_attr={"shape": "box"})
    view_cluster = Digraph("cluster_views",
                           graph_attr={"style": "solid", "label": "Views"}, node_attr={"shape": "box"})
    table_cluster = Digraph("cluster_tables",
                            graph_attr={"style": "solid", "rank": "same", "label": "Tables"},
                            node_attr={"shape": "box"})

    for cluster, nodes in {model_cluster: model_nodes, explore_cluster: explore_nodes}.items():
        for node in nodes:
            if node.path_key in manifest['child_map']:
                cluster.node(node.path_key)

    view_nodes_by_directory: dict[str, list[DepNode]] = {}

    for node in view_nodes:
        directory = node.directory_label(prefix_path)
        print(directory)

        if directory not in view_nodes_by_directory:
            view_nodes_by_directory[directory] = []

        view_nodes_by_directory[directory].append(node)

    folder_num = 0
    for directory, nodes in view_nodes_by_directory.items():
        folder_cluster = Digraph(f"cluster_views_{folder_num}_folder",
                                 graph_attr={"style": "dotted", "label": directory, "rankdir": "LR"},
                                 node_attr={"shape": "box"})

        folder_empty = True

        for node in nodes:
            if node.path_key in manifest['child_map']:
                folder_cluster.node(node.path_key)
                folder_empty = False

        if not folder_empty:
            view_cluster.subgraph(folder_cluster)

        folder_num += 1

    for project, nodes_by_dataset in table_nodes_by_project_by_dataset.items():
        project_cluster = Digraph("cluster_table_project_" + project,
                                  graph_attr={"style": "dashed", "rank": "same", "label": project},
                                  node_attr={"shape": "box"})

        project_empty = True

        for dataset, nodes in nodes_by_dataset.items():
            dataset_cluster = Digraph(f"cluster_table_project_{project}_datasets_{dataset}",
                                      graph_attr={"style": "dotted", "rank": "same", "label": dataset},
                                      node_attr={"shape": "box"})

            dataset_empty = True

            for node in nodes:
                if node.path_key in manifest['child_map']:
                    dataset_cluster.node(node.path_key)
                    dataset_empty = False

            if not dataset_empty:
                project_cluster.subgraph(dataset_cluster)
                project_empty = False

        if not project_empty:
            table_cluster.subgraph(project_cluster)

    g = Digraph("G", format=RENDER_FORMAT, node_attr={"color": "lightblue2", "style": "filled"},
                graph_attr={"newrank": "true", "rankdir": "LR", "concentrate": "true"})

    g.subgraph(model_cluster)
    g.subgraph(explore_cluster)
    g.subgraph(view_cluster)

    if not no_tables:
        g.subgraph(table_cluster)

    pairs = []
    for parent in manifest["child_map"].keys():
        for child in manifest["child_map"][parent]:
            pairs.append((parent, child))

    for pair in pairs:
        g.edge(*pair)

    return g


def render_graph(g: Digraph, path: Path):
    """Render a Graph saving output to a given path"""
    print(f"Rendering {path}")
    g.render(path, view=False)


@click.command()
@click.option(
    "--filters",
    help="Keep only edges connecting node passed. For multiple filters, pass a string with node names seperated by spaces",
)
@click.option(
    "--no-tables",
    is_flag=True,
    default=False,
    help="Render the tables section of the out",
)
def main(filters: str, no_tables: bool):
    nodes = Nodes()

    try:
        parsed_filters = []
        if filters:
            parsed_filters = filters.strip('\'"').split(" ")

        manifest = build_manifest(nodes, filters=parsed_filters, tables=not no_tables)

    except FileNotFoundError as e:
        print(repr(e))
        return

    g = build_graph(
        nodes,
        manifest,
        no_tables
    )

    filename = Path(OUTPUT_FILE_PATH)
    render_graph(g, filename)


if __name__ == "__main__":
    main()
