from typing import List, Dict, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data


def build_career_graph(
    transitions: List[Dict]
) -> Tuple[Data, Dict[str, int]]:
    """
    Build a PyTorch Geometric graph from career transitions.

    Args:
        transitions: List of dicts with keys:
            - from_job (str)
            - to_job (str)
            - years (float)

    Returns:
        data: torch_geometric.data.Data object
        job_mapping: dict mapping job title -> integer index
    """

    if not transitions:
        # Return empty graph
        return Data(), {}

    # Collect unique job titles
    jobs = set()
    for t in transitions:
        jobs.add(t["from_job"])
        jobs.add(t["to_job"])

    # Map job titles to contiguous indices
    job_mapping = {job: idx for idx, job in enumerate(sorted(jobs))}

    # Build edge index lists
    edge_sources = []
    edge_targets = []
    edge_years = []

    for t in transitions:
        src = job_mapping[t["from_job"]]
        dst = job_mapping[t["to_job"]]

        # Skip self-loops explicitly (documented decision)
        if src == dst:
            continue

        edge_sources.append(src)
        edge_targets.append(dst)
        edge_years.append([float(t["years"])])

    if not edge_sources:
        # All transitions were self-loops
        return Data(), job_mapping

    edge_index = torch.tensor(
        [edge_sources, edge_targets],
        dtype=torch.long
    )

    edge_attr = torch.tensor(
        edge_years,
        dtype=torch.float
    )

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data, job_mapping


def get_neighbors(data: Data, node_idx: int) -> List[int]:
    """
    Return all one-hop outgoing neighbors for a node.
    """

    if data.edge_index is None:
        return []

    # edge_index shape: [2, E]
    sources = data.edge_index[0]
    targets = data.edge_index[1]

    mask = sources == node_idx
    neighbors = targets[mask]

    return neighbors.tolist()