import asyncio
import os
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path

import httpx
import networkx as nx
import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from terminaltexteffects.effects import effect_beams

from dbt_cll_evaluator.cll_types import ColLineage, Model
from dbt_cll_evaluator.rules import (
    ConfigLineage,
    DocumentationDifference,
    ModelingUpstreamDownstream,
    Rename,
)

# uv run dbt_cll/main.py
app = typer.Typer()


@dataclass
class CLIState:
    dbt_cloud_api_token: str
    dbt_cloud_environment_id: str
    dbt_cloud_host_url: str
    start_dbt_node_unique_id: str
    export: bool
    debug: bool = False


def typer_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


MAX_DEPTH = None


async def get_cll(endpoint, api_token, env_id, node_unique_id, MAX_DEPTH) -> Model:
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_token}",
    }

    query = """query Lineage($nodeUniqueId: String!, $environmentId: BigInt!, $filters: ColumnLineageFilter) {
  column(environmentId: $environmentId) {
    lineage(nodeUniqueId: $nodeUniqueId, filters: $filters) {
      name
      uniqueId
      nodeUniqueId
      description
      descriptionOriginColumnName
      descriptionOriginResourceUniqueId
      transformationType
      relationship
      parentColumns
      childColumns
    }
  }
}
"""

    json_data = {
        "query": query,
        "variables": {
            "environmentId": env_id,
            "nodeUniqueId": node_unique_id,
            "filters": {
                "maxDepth": MAX_DEPTH,
            },
        },
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://metadata.{endpoint}/internal/graphql",
            headers=headers,
            json=json_data,
        )
        Path("response.json").write_text(response.text)
    return Model(**response.json())


async def generate_cll_info(
    endpoint, api_token, env_id, node_unique_id, MAX_DEPTH, debug=False
) -> Model:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=Console(force_terminal=not debug),
        transient=True,
    ) as progress:
        task_get_cll = progress.add_task(
            description="Getting Column Level Lineage info from the API...", total=None
        )
        response = await get_cll(endpoint, api_token, env_id, node_unique_id, MAX_DEPTH)
        progress.update(task_get_cll, description="CLL received", completed=True)

        progress.add_task(description="Building Graph...", total=None)

        all_details = response.data.column.lineage
        lineage_dict = {}
        G = nx.DiGraph()
        for lineage in all_details:
            lineage_dict[lineage.unique_id] = lineage

            G = build_column_dag_for_node(G, lineage)
    return G, lineage_dict, all_details


def build_column_dag_for_node(G: nx.DiGraph, lineage: ColLineage) -> nx.DiGraph:
    if 1:
        G.add_node(
            lineage.unique_id, full_lineage=True, node_unique_id=lineage.node_unique_id
        )
        for parent in lineage.parent_columns:
            if not G.has_edge(parent, lineage.unique_id):
                G.add_edge(
                    parent,
                    lineage.unique_id,
                    transformation_type=lineage.transformation_type,
                )
        for child in lineage.child_columns:
            if not G.has_edge(lineage.unique_id, child) and child[-1] != "*":
                G.add_edge(lineage.unique_id, child)
    return G


@app.command(help="Check for differences in the descriptions of passthrough columns")
@typer_async
async def desc_difference(
    ctx: typer.Context,
    pushdown_missing_desc: bool = typer.Option(
        False, help="Check for missing descriptions and push it down"
    ),
    dbt_project_path: str = typer.Option(
        "dummy",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Path to the root of the dbt project. The manifest needs to have been generated",
    ),
):
    if pushdown_missing_desc and dbt_project_path == "dummy":
        logger.error(
            "You need to provide a dbt project path to push down missing descriptions"
        )
        return

    G, lineage_dict, all_details = await generate_cll_info(
        ctx.obj.dbt_cloud_host_url,
        ctx.obj.dbt_cloud_api_token,
        ctx.obj.dbt_cloud_environment_id,
        ctx.obj.start_dbt_node_unique_id,
        MAX_DEPTH,
        ctx.obj.debug,
    )

    cll_diff = DocumentationDifference(
        lineage_dict, all_details, G, dbt_project_path=dbt_project_path
    )
    cll_diff.calculate()
    cll_diff.show()
    if ctx.obj.export:
        cll_diff.export()
    if pushdown_missing_desc:
        cll_diff.pushdown_missing_desc()


@app.command(help="Check when columns are renamed outside of allowed models")
@typer_async
async def rename(ctx: typer.Context):
    G, lineage_dict, all_details = await generate_cll_info(
        ctx.obj.dbt_cloud_host_url,
        ctx.obj.dbt_cloud_api_token,
        ctx.obj.dbt_cloud_environment_id,
        ctx.obj.start_dbt_node_unique_id,
        MAX_DEPTH,
        ctx.obj.debug,
    )

    cll_rename = Rename(lineage_dict, all_details)
    cll_rename.calculate()
    cll_rename.show()
    if ctx.obj.export:
        cll_rename.export()


@app.command(
    help="Check for the number of upstream and downstream dependencies for columns"
)
@typer_async
async def num_edges(ctx: typer.Context):
    G, lineage_dict, all_details = await generate_cll_info(
        ctx.obj.dbt_cloud_host_url,
        ctx.obj.dbt_cloud_api_token,
        ctx.obj.dbt_cloud_environment_id,
        ctx.obj.start_dbt_node_unique_id,
        MAX_DEPTH,
        ctx.obj.debug,
    )

    cll_num_edges = ModelingUpstreamDownstream(G)
    cll_num_edges.calculate()
    cll_num_edges.show()
    if ctx.obj.export:
        cll_num_edges.export()


class ConfigCheck(str, Enum):
    tags = "tags"
    meta = "meta"


@app.command(
    help="Check that specific config is present for all upstream and downstream columns"
)
@typer_async
async def config_lineage(
    ctx: typer.Context,
    key: ConfigCheck = typer.Option(..., help="What config type to check"),
    value: str = typer.Option(..., help="Value to check for"),
    fix: bool = typer.Option(
        False,
        help="Fix the YML files",
    ),
    dbt_project_path: str = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Path to the root of the dbt project. The manifest needs to have been generated",
    ),
):
    if key.value == "meta":
        logger.error("'meta' is not implemented yet, we only support 'tags' for now")
        return
    if fix and dbt_project_path is None:
        logger.error("You need to provide a dbt project path to fix the YML files")
        return

    G, lineage_dict, all_details = await generate_cll_info(
        ctx.obj.dbt_cloud_host_url,
        ctx.obj.dbt_cloud_api_token,
        ctx.obj.dbt_cloud_environment_id,
        ctx.obj.start_dbt_node_unique_id,
        MAX_DEPTH,
        ctx.obj.debug,
    )

    cll_config_lineage = ConfigLineage(
        lineage_dict, all_details, G, key.value, value, dbt_project_path
    )
    cll_config_lineage.calculate()
    cll_config_lineage.show()
    if fix:
        cll_config_lineage.fix()
    if ctx.obj.export:
        cll_config_lineage.export()


@app.command(help="Experimental -- Check for columns being re-joined downstream")
@typer_async
async def exp_rejoin(ctx: typer.Context):
    G, lineage_dict, all_details = await generate_cll_info(
        ctx.obj.dbt_cloud_host_url,
        ctx.obj.dbt_cloud_api_token,
        ctx.obj.dbt_cloud_environment_id,
        ctx.obj.start_dbt_node_unique_id,
        MAX_DEPTH,
        ctx.obj.debug,
    )

    nodes_with_multiple_paths = set()

    source_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    target_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]

    for source in track(source_nodes, "Looping through nodes", disable=ctx.obj.debug):
        for target in target_nodes:
            if source != target and "days" not in source:
                paths = list(nx.all_simple_paths(G, source=source, target=target))
                if len(paths) >= 2:
                    nodes_with_multiple_paths.add(f"{source} -> {target}")
    print(nodes_with_multiple_paths)


@app.callback()
def common(
    ctx: typer.Context,
    dbt_cloud_api_token=typer.Option(
        ...,
        envvar="DBT_CLOUD_TOKEN",
        help="dbt Cloud API Token to connect to the Metadata API",
    ),
    dbt_cloud_environment_id=typer.Option(
        ...,
        envvar="DBT_CLOUD_ENV_ID",
        help="dbt Cloud environment ID for the dbt Cloud project we are interested in",
    ),
    dbt_cloud_host_url=typer.Option(
        "cloud.getdbt.com",
        envvar="DBT_CLOUD_HOST_URL",
        help="Host URL for the dbt Cloud account",
    ),
    start_unique_id=typer.Option(
        ...,
        envvar="DBT_CLL_START_NODE",
        help="Node Unique ID to start the lineage from. In the format `model.<project_name>.<model_name>`",
    ),
    export: bool = typer.Option(
        False,
        envvar="DBT_CLL_EXPORT",
        help="Whether we want to export the data as CSV or not",
    ),
    debug: bool = typer.Option(
        False, envvar="DBT_CLL_DEBUG", help="Debug mode with less animations"
    ),
):
    """
    Handle global flags.
    """
    ctx.obj = CLIState(
        dbt_cloud_api_token=dbt_cloud_api_token,
        dbt_cloud_environment_id=dbt_cloud_environment_id,
        dbt_cloud_host_url=dbt_cloud_host_url,
        start_dbt_node_unique_id=start_unique_id,
        export=export,
        debug=debug,
    )


def center_multiline_string(multiline_str, width):
    lines = multiline_str.splitlines()
    centered_lines = [line.center(width) for line in lines]
    return "\n".join(centered_lines)


def startup():
    if os.environ.get("DBT_CLL_UI_EFFECTS") is not None:
        effect = effect_beams.Beams(
            center_multiline_string(
                "Live from Coalesce 2024!\ndbt-cll-evaluator\nMade with ❤️ (and maybe bugs) as Open Source Software",
                60,
            ),
        )
        effect.effect_config.beam_delay = 5
        with effect.terminal_output() as terminal:
            for frame in effect:
                terminal.print(frame)
        time.sleep(2)

    app()


if __name__ == "__main__":
    startup()


# - column has the same name when it is passthrough
# - column descriptions are the same when it is passthrough
# - if downstream column description is different, change it
# - if downstream/upstream column description is not there, add it
# - columns with the same name need to have the same upstream dependencies
# - rename/passthrough but with more than 1 column
