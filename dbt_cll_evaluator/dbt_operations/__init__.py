import json
from pathlib import Path

import networkx as nx
from loguru import logger
from ruamel.yaml import YAML


class DbtOps:
    def __init__(self, dbt_project_path):
        self.dbt_project_path = dbt_project_path
        with (Path(dbt_project_path) / "target/manifest.json").open("r") as file:
            self.dbt_info = json.load(file)

    def has_column_config(self, G: nx.DiGraph, column_id, field, value) -> bool:
        if field == "tags":
            node_id = G.nodes[column_id].get("node_unique_id")
            column_name = column_id.split(".")[-1]
            if node_id in self.dbt_info["nodes"]:
                column_tags = (
                    self.dbt_info["nodes"]
                    .get(node_id, {})
                    .get("columns", {})
                    .get(column_name, {})
                    .get("tags", [])
                )
                if value in column_tags:
                    return True
            if node_id in self.dbt_info["sources"]:
                column_tags = (
                    self.dbt_info["sources"]
                    .get(node_id, {})
                    .get("columns", {})
                    .get(column_name, {})
                    .get("tags", [])
                )
                if value in column_tags:
                    return True
        return False

    def get_columns_with_config(self, G: nx.DiGraph, field, value) -> list[str]:
        matching_columns = []
        for node in G.nodes():
            if self.has_column_config(G, node, field, value):
                matching_columns.append(node)
        return matching_columns

    def update_column_config(self, G: nx.DiGraph, column_id, field, value) -> None:
        node_id = G.nodes[column_id].get("node_unique_id")
        model_name = node_id.split(".")[-1]
        column_name = column_id.split(".")[-1]

        if field == "tags":
            yml_patch_path = None
            if node_id in self.dbt_info["nodes"]:
                yml_patch_path = self.dbt_info["nodes"][node_id].get("patch_path")
                yml_node_name = "models"

            elif node_id in self.dbt_info["sources"]:
                # TODO: Implement this
                pass

            if yml_patch_path:
                # HACK and needs to be fixed. Doesn't work with packages
                yml_file = Path(self.dbt_project_path) / yml_patch_path.split("//")[1]
                if yml_file is None:
                    logger.error(f"Could not find YML file for the node {node_id}")
                    return
                else:
                    with open(yml_file, "r") as file:
                        yml = YAML()
                        yml.default_flow_style = False
                        yml.preserve_quotes = True
                        yml.width = 120
                        yml.indent(mapping=2, sequence=4, offset=2)
                        yml_data = yml.load(file)

                    # model_info = [(idx, model) for (idx,model) in yml_data["models"].enumerate() if model["name"] == model_name]
                    node_info = [
                        node
                        for node in yml_data[yml_node_name]
                        if node["name"] == model_name
                    ]
                    if not node_info:
                        logger.warning(
                            f"Could not find model {model_name} in the YML file"
                        )
                        return

                    # check if there is a column config overall
                    if "columns" not in node_info[0]:
                        node_info[0]["columns"] = []

                    column_config = [
                        column
                        for column in node_info[0]["columns"]
                        if column["name"] == column_name
                    ]
                    if not column_config:
                        logger.info(
                            f"Could not find column {column_name} in the YML file {yml_file} for model {model_name}, creating it"
                        )
                        node_info[0]["columns"].append(
                            {"name": column_name, "tags": [value]}
                        )

                    else:
                        if "tags" not in column_config[0]:
                            column_config[0]["tags"] = []
                        # we need to check if is not already there
                        # this could happen because it was added recently and the manifest was not updated
                        if value in column_config[0]["tags"]:
                            logger.info(
                                f"The YML file {yml_file} already has the tag {value} for column {column_name} in model {model_name}. The manifest is not up to date, please update it to avoid issues."
                            )
                            return
                        column_config[0]["tags"].append(value)

                    with open(yml_file, "w") as file:
                        yml.dump(yml_data, file)
                        logger.success(
                            f"Updated the YML file {yml_file} with the tag {value} for column {column_name} in model {model_name}"
                        )

            else:
                logger.warning(f"Could not find node_id {node_id} in the dbt manifest")

    def add_description(self, G: nx.DiGraph, column_id, description) -> None:
        node_id = G.nodes[column_id].get("node_unique_id")
        model_name = node_id.split(".")[-1]
        column_name = column_id.split(".")[-1]

        yml_patch_path = self.dbt_info["nodes"][node_id].get("patch_path")
        yml_node_name = "models"

        if yml_patch_path:
            # HACK and needs to be fixed. Doesn't work with packages
            yml_file = Path(self.dbt_project_path) / yml_patch_path.split("//")[1]
            if yml_file is None:
                logger.error(f"Could not find YML file for the node {node_id}")
                return
            else:
                with open(yml_file, "r") as file:
                    yml = YAML()
                    yml.default_flow_style = False
                    yml.preserve_quotes = True
                    yml.width = 120
                    yml.indent(mapping=2, sequence=4, offset=2)
                    yml_data = yml.load(file)

                # model_info = [(idx, model) for (idx,model) in yml_data["models"].enumerate() if model["name"] == model_name]
                node_info = [
                    node
                    for node in yml_data[yml_node_name]
                    if node["name"] == model_name
                ]
                if not node_info:
                    logger.warning(f"Could not find model {model_name} in the YML file")
                    return

                # check if there is a column config overall
                if "columns" not in node_info[0]:
                    node_info[0]["columns"] = []

                column_info = [
                    column
                    for column in node_info[0]["columns"]
                    if column["name"] == column_name
                ]
                if not column_info:
                    logger.info(
                        f"Could not find column {column_name} in the YML file {yml_file} for model {model_name}, creating it"
                    )
                    node_info[0]["columns"].append(
                        {"name": column_name, "description": description}
                    )
                elif column_info[0].get("description", "") != "":
                    logger.info(
                        f"The YML file {yml_file} already has a description for column {column_name} in model {model_name}. We are not updating it."
                    )
                    return
                else:
                    column_info[0]["description"] = description

                with open(yml_file, "w") as file:
                    yml.dump(yml_data, file)
                    logger.success(
                        f"Updated the YML file {yml_file} with the upstream description for column {column_name} in model {model_name}"
                    )

        else:
            logger.warning(f"Could not find node_id {node_id} in the dbt manifest")
