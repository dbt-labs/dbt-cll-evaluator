import csv
from dataclasses import asdict, dataclass, fields
from typing import List, Protocol

import networkx as nx
from rich.console import Console
from rich.table import Table

from dbt_cll_evaluator.cll_types import ColLineage
from dbt_cll_evaluator.dbt_operations import DbtOps


def allow_rename(node_unique_id: str) -> bool:
    return node_unique_id.split(".")[-1].startswith("stg_") or node_unique_id.split(
        "."
    )[-1].startswith("base_")


def is_root_model(col_unique_id: str) -> bool:
    return col_unique_id.split(".")[0] == "raw"


def is_final_model(col_unique_id: str) -> bool:
    return col_unique_id.split(".")[-2].startswith("fct_") or col_unique_id.split(".")[
        -2
    ].startswith("dim_")


def export_dataclass_to_csv(data_list: List[dataclass], file_name: str) -> None:
    if not data_list:
        return  # No data to write

    fieldnames = [f.name for f in fields(data_list[0])]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_list:
            writer.writerow(asdict(data))


class CLLRule(Protocol):
    def calculate(self) -> None: ...

    def show(self) -> None: ...

    def export(self) -> None: ...


class DocumentationDifference(CLLRule):
    @dataclass
    class Data:
        upstream_id: str
        upstream_desc: str
        downstream_id: str
        downstream_desc: str
        comment: str

    def __init__(
        self,
        lineage_dict: dict,
        all_lineage: List[ColLineage],
        G: nx.DiGraph,
        dbt_project_path=None,
    ) -> None:
        self.lineage_dict = lineage_dict
        self.all_lineage = all_lineage
        self.G = G
        self.dbt_project_path = dbt_project_path
        self.list_difference: List[DocumentationDifference.Difference] = []

    def calculate(self) -> None:
        for lineage in self.all_lineage:
            if lineage.transformation_type == "PASSTHROUGH":
                desc_casefold = (
                    lineage.description.casefold().strip()
                    if lineage.description
                    else None
                )
                # sometimes the upstream node is not there... so we return nothing but we should make it better

                if not lineage.parent_columns:
                    upstream_desc = None
                else:
                    lineage_parent_column = lineage.parent_columns[0]
                    if lineage_parent_column in self.lineage_dict:
                        upstream_desc = self.lineage_dict[
                            lineage_parent_column
                        ].description
                    else:
                        upstream_desc = None

                upstream_casefold = (
                    upstream_desc.casefold().strip() if upstream_desc else None
                )

                if (
                    desc_casefold
                    and upstream_casefold
                    and desc_casefold != upstream_casefold
                ):
                    self.list_difference.append(
                        DocumentationDifference.Data(
                            upstream_id=lineage.parent_columns[0],
                            upstream_desc=upstream_desc,
                            downstream_id=lineage.unique_id,
                            downstream_desc=lineage.description,
                            comment="Description is different",
                        )
                    )

                if (
                    upstream_casefold
                    and lineage.description_origin_column_name is not None
                ):
                    self.list_difference.append(
                        DocumentationDifference.Data(
                            upstream_id=lineage.parent_columns[0],
                            upstream_desc=upstream_desc,
                            downstream_id=lineage.unique_id,
                            downstream_desc=None,
                            comment="Description is missing downstream",
                        )
                    )

            if lineage.transformation_type == "RAW":
                pass

            if lineage.transformation_type == "TRANSFORMATION":
                pass

    def pushdown_missing_desc(self) -> None:
        for diff in self.list_difference:
            if diff.downstream_desc is None:
                dbt_ops = DbtOps(self.dbt_project_path)
                dbt_ops.add_description(self.G, diff.downstream_id, diff.upstream_desc)

    def show(self) -> None:
        table = Table(title="Different descriptions", row_styles=["none", "dim"])

        table.add_column("upstream_id")
        table.add_column("upstream_desc")
        table.add_column("downstream_id")
        table.add_column("downstream_desc")
        table.add_column("comment")

        for diff in self.list_difference:
            table.add_row(
                diff.upstream_id,
                diff.upstream_desc,
                diff.downstream_id,
                diff.downstream_desc,
                diff.comment,
            )

        console = Console()
        console.print(table)

    def export(self) -> None:
        export_dataclass_to_csv(
            self.list_difference, "columns_documentation_difference.csv"
        )


class Rename(CLLRule):
    @dataclass
    class Data:
        upstream_id: str
        upstream_name: str
        downstream_id: str
        downstream_name: str
        comment: str

    def __init__(self, lineage_dict: dict, all_lineage: List[ColLineage]) -> None:
        self.lineage_dict = lineage_dict
        self.all_lineage = all_lineage
        self.list_renames: List[DocumentationDifference.Difference] = []

    def calculate(self) -> None:
        for lineage in self.all_lineage:
            if lineage.transformation_type == "PASSTHROUGH":
                pass

            if lineage.transformation_type == "RENAME":
                if not allow_rename(lineage.node_unique_id):
                    if len(lineage.parent_columns) == 1:
                        self.list_renames.append(
                            Rename.Data(
                                upstream_id=lineage.parent_columns[0],
                                upstream_name=self.lineage_dict[
                                    lineage.parent_columns[0]
                                ].name,
                                downstream_id=lineage.unique_id,
                                downstream_name=lineage.name,
                                comment="Column is renamed in a non-stg model",
                            )
                        )
                    else:
                        self.list_renames.append(
                            Rename.Data(
                                upstream_id=";".join(lineage.parent_columns),
                                upstream_name=None,
                                downstream_id=lineage.unique_id,
                                downstream_name=lineage.name,
                                comment="Column is renamed from multiple upstream columns",
                            )
                        )

            if lineage.transformation_type == "RAW":
                pass

            if lineage.transformation_type == "TRANSFORMATION":
                pass

    def show(self) -> None:
        table = Table(title="Column renamed", row_styles=["none", "dim"])

        table.add_column("upstream_id")
        table.add_column("upstream_name")
        table.add_column("downstream_id")
        table.add_column("downstream_name")
        table.add_column("comment")

        for diff in self.list_renames:
            table.add_row(
                diff.upstream_id,
                diff.upstream_name,
                diff.downstream_id,
                diff.downstream_name,
                diff.comment,
            )

        console = Console()
        console.print(table)

    def export(self) -> None:
        export_dataclass_to_csv(self.list_renames, "columns_rename.csv")


class ModelingUpstreamDownstream(CLLRule):
    @dataclass
    class Data:
        model_name: str
        column_name: str
        num_edges: int
        comment: str

    def __init__(self, G: nx.DiGraph, limit_lots_dependencies: int = 10) -> None:
        self.G = G
        self.list_upstream: List[ModelingUpstreamDownstream.Data] = []
        self.list_downstream: List[ModelingUpstreamDownstream.Data] = []
        self.limit_lots_dependencies = limit_lots_dependencies

    def calculate(self) -> None:
        in_degrees = self.G.in_degree()
        for node, in_degree in in_degrees:
            # for some reason, sometimes we list the model name and a *, we don't want to count those
            if node.split(".")[-1] == "*":
                continue

            if self.G.nodes[node].get("full_lineage"):
                message = ""
                if in_degree == 0 and not is_root_model(node):
                    message = "No upstream edges but not a root"
                if in_degree > self.limit_lots_dependencies:
                    message = "There are many upstream edges, you should have a unit test to capture all edge cases"

                self.list_upstream.append(
                    ModelingUpstreamDownstream.Data(
                        model_name=".".join(node.split(".")[:-1]),
                        column_name=node.split(".")[-1],
                        num_edges=in_degree,
                        comment=message,
                    )
                )

        out_degrees = self.G.out_degree()
        for node, out_degree in out_degrees:
            # for some reason, sometimes we list the model name and a *, we don't want to count those
            if node.split(".")[-1] == "*":
                continue

            if self.G.nodes[node].get("full_lineage"):
                message = ""
                if out_degree == 0 and not is_final_model(node):
                    message = "No downstream edges but not a final model"
                if out_degree > self.limit_lots_dependencies:
                    message = "There are many downstrean edges, this column is critical, make sure it is well tested"

                self.list_downstream.append(
                    ModelingUpstreamDownstream.Data(
                        model_name=".".join(node.split(".")[:-1]),
                        column_name=node.split(".")[-1],
                        num_edges=out_degree,
                        comment=message,
                    )
                )

    def show(self) -> None:
        table_upstream = Table(
            title="Upstream edges analysis", row_styles=["none", "dim"]
        )

        table_upstream.add_column("model_name")
        table_upstream.add_column("column_name")
        table_upstream.add_column("num_upstream_edges")
        table_upstream.add_column("comment")

        for diff in sorted(self.list_upstream, key=lambda x: x.model_name):
            table_upstream.add_row(
                diff.model_name, diff.column_name, str(diff.num_edges), diff.comment
            )

        table_downstream = Table(
            title="Downstream edges analysis", row_styles=["none", "dim"]
        )

        table_downstream.add_column("model_name")
        table_downstream.add_column("column_name")
        table_downstream.add_column("num_downstram_edges")
        table_downstream.add_column("comment")

        for diff in sorted(self.list_downstream, key=lambda x: x.model_name):
            table_downstream.add_row(
                diff.model_name, diff.column_name, str(diff.num_edges), diff.comment
            )

        console = Console()
        console.print(table_upstream)
        console.print(table_downstream)

    def export(self) -> None:
        export_dataclass_to_csv(
            self.list_downstream, "columns_downstream_dependencies.csv"
        )
        export_dataclass_to_csv(self.list_upstream, "columns_upstream_dependencies.csv")


class ConfigLineage(CLLRule):
    @dataclass
    class Data:
        column_with_config: str
        column_without_config: str
        model_without_config: str
        comment: str
        key: str
        value: str

    def __init__(
        self,
        lineage_dict,
        all_details,
        G: nx.DiGraph,
        key,
        value,
        dbt_project_path=None,
    ) -> None:
        self.G = G
        self.lineage_dict = lineage_dict
        self.all_details = all_details
        self.G = G
        self.key = key
        self.value = value
        self.dbt_project_path = dbt_project_path
        self.missing_config: List[ConfigLineage.Data] = []

    def calculate(self) -> None:
        dbt_ops = DbtOps(self.dbt_project_path)
        columns_with_config = dbt_ops.get_columns_with_config(
            self.G, self.key, self.value
        )

        for column in columns_with_config:
            upstream_columns = self.G.predecessors(column)
            for upstream_column in upstream_columns:
                if upstream_column not in columns_with_config:
                    self.missing_config.append(
                        ConfigLineage.Data(
                            column_with_config=column,
                            column_without_config=upstream_column,
                            model_without_config=self.G.nodes[upstream_column].get(
                                "node_unique_id"
                            ),
                            comment=f"Upstream column is missing the config {self.key}: {self.value}",
                            key=self.key,
                            value=self.value,
                        )
                    )

            downstream_columns = self.G.successors(column)
            for downstream_column in downstream_columns:
                if downstream_column not in columns_with_config:
                    self.missing_config.append(
                        ConfigLineage.Data(
                            column_with_config=column,
                            column_without_config=downstream_column,
                            model_without_config=self.G.nodes[downstream_column].get(
                                "node_unique_id"
                            ),
                            comment=f"Downstream column is missing the config {self.key}: {self.value}",
                            key=self.key,
                            value=self.value,
                        )
                    )

    def fix(self) -> None:
        dbt_ops = DbtOps(self.dbt_project_path)

        for conf in self.missing_config:
            dbt_ops.update_column_config(
                self.G, conf.column_without_config, conf.key, conf.value
            )

    def show(self) -> None:
        table = Table(title="Missing config", row_styles=["none", "dim"])

        table.add_column("column_with_config")
        table.add_column("column_without_config")
        table.add_column("model_without_config")
        table.add_column("comment")

        for conf in self.missing_config:
            table.add_row(
                conf.column_with_config,
                conf.column_without_config,
                conf.model_without_config,
                conf.comment,
            )

        console = Console()
        console.print(table)

    def export(self) -> None:
        export_dataclass_to_csv(self.missing_config, "columns_missing_config.csv")
