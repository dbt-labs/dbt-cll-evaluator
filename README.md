# dbt-cll-evaluator

CLI to interact with the column-level lineage calculated by dbt Cloud and visible in dbt Explorer.
Presented at Coalesce 2024.

⚠️ This tool leverages some internal API, and there is no guarantee that it will work forever ⚠️

## Coalesce 2024 presentation

Here are links to the [slides](<slides/CLL Coalesce 24.pdf>) and to [the recording](https://www.getdbt.com/resources/coalesce-on-demand/coalesce-2024-leveraging-column-level-lineage-to-scale-your-dbt-projects).

## Installing it and running it

- clone this repo
- install `poetry` 
- type `poetry install` to install the relevant dependencies
- type `poetry run dbt-cll-evaluator ...` to use the tool

`poetry run dbt-cll-evaluator --help` will list the different options available in the tool

## Useful information

All commands require you to provide a dbt Cloud API token with access to the Discovery API.
You can use a [Personal Access Token](https://docs.getdbt.com/docs/dbt-cloud-apis/user-tokens#account-scoped-personal-access-tokens) or a [Service Token](https://docs.getdbt.com/docs/dbt-cloud-apis/service-tokens).

We also need to provide the dbt Cloud environment ID from which we want to retrieve the information. You will likely want to pick the Production environment ID of the dbt Cloud project you are interested in.

Finally, today, the API returns the column-level linage information for all columns upstream and downstream of a give model (e.g. it returns the CLL for all the `+my_model+`). This means that we currently can't get the entire column level lineage at once and we need to provide a `node_id` to start from (e.g. `model.my_dbt_project.my_model`).

All the commands that support a `--fix` option will also require you to provide the path where your dbt project is located.
