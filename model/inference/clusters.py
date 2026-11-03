"""Group adjacent fire pixels: one fire front usually covers several 2 km pixels."""

import numpy as np


def clusters(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Groups of 8-connected True pixels. The masks here are sparse, so a search over
    the True pixels is enough."""
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        stack = [remaining.pop()]
        group = []
        while stack:
            row, col = stack.pop()
            group.append((row, col))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (row + dy, col + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        groups.append(group)
    return groups
