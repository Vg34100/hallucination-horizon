from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx

Coord = Tuple[int, int]


def build_graph(width: int, height: int, walls: Iterable[Coord]) -> nx.Graph:
    wall_set = set(walls)
    g = nx.Graph()
    for y in range(height):
        for x in range(width):
            if (x, y) in wall_set:
                continue
            g.add_node((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < width and 0 <= ny_ < height and (nx_, ny_) not in wall_set:
                    g.add_edge((x, y), (nx_, ny_))
    return g


def astar_path(width: int, height: int, walls: Iterable[Coord], start: Coord, goal: Coord) -> List[Coord]:
    g = build_graph(width, height, walls)

    def heuristic(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    return nx.astar_path(g, start, goal, heuristic=heuristic)


def all_pairs_shortest_lengths(width: int, height: int, walls: Iterable[Coord]) -> Dict[Coord, Dict[Coord, int]]:
    g = build_graph(width, height, walls)
    return dict(nx.all_pairs_shortest_path_length(g))
