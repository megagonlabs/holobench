import json
from collections import Counter, defaultdict
from typing import Dict, List, Union

import duckdb
import pandas as pd
from datasets import DatasetDict, load_dataset
from sql_metadata import Parser

# Constant ratio to convert characters to tokens in LLMs.
CHAR_TO_TOKEN_RATIO = 3.66


def merge_contexts(
    main_context: List[str],
    auxiliary_context: List[str],
    merge_strategy: str = "uniform",
) -> List[str]:
    """
    Merges two lists of context strings (`main_context` and `auxiliary_context`) according to a specified merging strategy.

    Parameters:
    -----------
    main_context : List[str]
        The primary context that takes priority in the merging process.
    auxiliary_context : List[str]
        The secondary context to be merged with the main context.
    merge_strategy : str, optional
        The strategy used for merging the two contexts. Options include:
        - "uniform" (default): Inserts auxiliary context uniformly into the main context.
        - "begin": Main context is appended to the beginning of the auxiliary context.
        - "end": Main context is appended to the end of the auxiliary context.
        - "middle": Inserts main context into the middle of the auxiliary context.
        - "bimodal": Splits mains context into two halves and inserts them at the beginning and end of the auxiliary context.

    Returns:
    --------
    List[str]
        The merged context as a list of strings, following the selected merge strategy.

    """
    if min(map(len, (main_context, auxiliary_context))) == 0:
        return main_context + auxiliary_context

    merged_context = []
    if merge_strategy == "uniform":
        merged_context = main_context[:]
        total_length = len(main_context) + len(auxiliary_context)
        step_size = total_length / len(auxiliary_context)
        indices = [int(i * step_size) for i in range(len(auxiliary_context))]
        for index, item in zip(indices, auxiliary_context):
            if index >= len(merged_context):
                merged_context.append(item)
            else:
                merged_context.insert(index, item)
    elif merge_strategy == "begin":
        merged_context = main_context + auxiliary_context
    elif merge_strategy == "end":
        merged_context = auxiliary_context + main_context
    elif merge_strategy == "middle":
        mid_point = len(main_context) // 2
        merged_context = (
            auxiliary_context[:mid_point] + main_context + auxiliary_context[mid_point:]
        )
    elif merge_strategy == "bimodal":
        mid_point = len(auxiliary_context) // 2
        merged_context = (
            main_context[:mid_point] + auxiliary_context + main_context[mid_point:]
        )
    else:
        raise ValueError()

    return merged_context


def execute_sql_query(query: str, table_data: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Executes an SQL query against the provided table data using DuckDB in-memory database.

    Parameters:
    -----------
    query : str
        The SQL query to be executed.
    table_data : Dict[str, List[Dict]]
        A dictionary where the key is the table name and the value is a list of dictionaries representing rows of data.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the result of the SQL query execution.
    """
    # Open an in-memory DuckDB connection
    with duckdb.connect(":memory:") as conn:
        # Register each table's data with the DuckDB instance
        for table_name, rows in table_data.items():
            conn.register(table_name, pd.DataFrame(rows))

        # Execute the SQL query and return the result as a pandas DataFrame
        return conn.query(query).df()


def load_context(
    query: str,
    db: Union[DatasetDict, str],
    max_context_size: int,
    info_density: float = 1.0,
    info_amount: int = -1,
    merge_strategy: str = "uniform",
):
    """
    Loads context from a dataset based on an SQL query and constraints like maximum context size and information density.

    Parameters:
    -----------
    query : str
        The SQL query which will be used to extract data from the provided database.
    db : Union[DatasetDict, str]
        A DatasetDict object or a string representing the dataset to be loaded. If a string is provided, the dataset is fetched from `megagonlabs/holobench`.
    max_context_size : int
        The maximum number of tokens allowed for the context.
    info_density : float, optional
        A ratio (between 0 and 1) indicating the density of information to include from the main context. Defaults to 1.0 (full density).
    info_amount : int, optional
        The maximum amount of information to extract from the main context, in tokens. Defaults to -1. If set to -1, `info_density` is used to determine the amount of information.
    merge_strategy : str, optional
        The strategy for merging main and auxiliary contexts. Defaults to "uniform". See `merge_contexts` function for available strategies.

    Returns:
    --------
    Tuple[str, pd.DataFrame]
        A tuple containing:
        - The combined context as a single string.
        - The result of the SQL query as a pandas DataFrame.

    Raises:
    -------
    ValueError
        If `info_amount` exceeds `max_context_size`.
    """

    # If db is a string, load the dataset.
    if isinstance(db, str):
        db = load_dataset("megagonlabs/holobench", name=db)

    # Determine the maximum size for the main context based on info_amount or info_density
    if info_amount > 0:
        max_main_size = info_amount
    else:
        max_main_size = int(max_context_size * info_density)

    # Calculate auxiliary size by subtracting the amount allocated to the main context
    max_auxiliary_size = max_context_size - info_amount

    # Calculate the maximum size of tables in the dataset
    max_table_size = max(map(len, db.values()))

    # Extract the table names involved in the SQL query
    main_tables = set(Parser(query).tables)

    # Initialize variables to hold context data and their respective sizes
    contexts, context_size = defaultdict(list), Counter()

    sub_db = defaultdict(list)

    # Iterating through the tables to gather context
    while True:
        for i in range(max_table_size):
            for table_name in db:
                is_main = table_name in main_tables
                threshold = max_main_size if is_main else max_auxiliary_size

                table = db[table_name]
                if i >= len(table) or context_size[is_main] >= threshold:
                    continue

                # Extract the text and calculate its token size
                entry = table[i]
                text = entry["verbalized_text"]
                context_size[is_main] += (
                    len(text) // CHAR_TO_TOKEN_RATIO + 1
                )  # +1 for whitespace

                # If within the threshold, add to the context
                if context_size[is_main] < threshold:
                    contexts[is_main].append(text)
                    if is_main:
                        sub_db[table_name].append(json.loads(entry["table_row"]))

            # Stop gathering context once thresholds are reached
            if (
                context_size[True] >= max_main_size
                and context_size[False] >= max_auxiliary_size
            ):
                # Merge main and auxiliary contexts using the selected strategy
                contexts = merge_contexts(
                    contexts[True], contexts[False], merge_strategy
                )

                # Join all context segments into a single string
                context = "\n".join(contexts)

                # Execute the SQL query on the gathered data
                gold_output = execute_sql_query(query, sub_db)

                return context, gold_output
