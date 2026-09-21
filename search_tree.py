import heapq
import itertools
import sys
from typing import Callable, FrozenSet, List, NamedTuple, Optional, Set, Tuple

from treelib import Tree

from halma import check_legal_move, initial_pos_1v1, win_cells_1v1

BOARD_SIZE = 5
PLAYER = 1

Square = Tuple[int, int]
State = FrozenSet[Square]
Move = Tuple[str, State]

START: State = frozenset(
    (row, column)
    for row in range(BOARD_SIZE)
    for column in range(BOARD_SIZE)
    if initial_pos_1v1[row][column] == PLAYER
)
GOAL: State = frozenset(win_cells_1v1[PLAYER])

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def square_name(square: Square) -> str:
    row, column = square
    return chr(ord("A") + column) + str(row + 1)


def to_board(state: State) -> List[List[int]]:
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for row, column in state:
        board[row][column] = PLAYER
    return board


def in_bounds(square: Square) -> bool:
    return 0 <= square[0] < BOARD_SIZE and 0 <= square[1] < BOARD_SIZE


def successors(state: State) -> List[Move]:
    board = to_board(state)
    moves: List[Move] = []
    for piece in sorted(state):
        for d_row, d_column in DIRECTIONS:
            for distance in (1, 2):
                target = (piece[0] + d_row * distance, piece[1] + d_column * distance)
                if in_bounds(target) and check_legal_move(board, piece, target):
                    label = f"{square_name(piece)}->{square_name(target)}"
                    moves.append((label, (state - {piece}) | {target}))
    return moves


def zero_heuristic(state: State) -> int:
    return 0


def goal_distance_heuristic(state: State) -> int:
    pieces = sorted(state)
    return min(
        sum(
            (abs(piece[0] - goal[0]) + abs(piece[1] - goal[1]) + 1) // 2
            for piece, goal in zip(pieces, assignment)
        )
        for assignment in itertools.permutations(sorted(GOAL))
    )


class SearchResult(NamedTuple):
    tree: Tree
    goal: Optional[State]
    expanded: int
    generated: int
    pruned: int


def build_tree(heuristic: Callable[[State], int] = zero_heuristic) -> SearchResult:
    tree = Tree()
    counter = itertools.count()
    expanded = generated = pruned = 0

    # heap entries: (priority, heuristic, tie-break, state, parent, move label, depth)
    frontier = [(heuristic(START), heuristic(START), next(counter), START, None, "START", 0)]

    while frontier:
        _, _, _, state, parent, label, depth = heapq.heappop(frontier)
        if state in tree:
            pruned += 1
            continue

        tree.create_node(label, state, parent=parent, data=depth)
        expanded += 1
        if state == GOAL:
            return SearchResult(tree, state, expanded, generated, pruned)

        for move_label, child in successors(state):
            generated += 1
            if child in tree:
                pruned += 1
                continue
            h = heuristic(child)
            heapq.heappush(
                frontier,
                (depth + 1 + h, h, next(counter), child, state, move_label, depth + 1),
            )

    return SearchResult(tree, None, expanded, generated, pruned)


def shortest_path(result: SearchResult) -> List[str]:
    if result.goal is None:
        return []
    ids = list(result.tree.rsearch(result.goal))[::-1]
    return [result.tree[i].tag for i in ids[1:]]


def highlight_path(result: SearchResult) -> Set[State]:
    on_path = set(result.tree.rsearch(result.goal)) if result.goal else set()
    for node_id in on_path:
        node = result.tree[node_id]
        node.tag = f"**{node.tag}**"
    return on_path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("start:", sorted(square_name(s) for s in START))
    print("goal: ", sorted(square_name(s) for s in GOAL))

    print(f"heuristic at start = {goal_distance_heuristic(START)}")

    runs = {
        "bfs": ("BFS, no ordering", build_tree()),
        "ordered": ("A*, ordered by heuristic", build_tree(goal_distance_heuristic)),
    }
    print(f"\n{'search':28}{'moves':>7}{'tree nodes':>12}{'generated':>11}{'pruned':>9}")
    for name, (title, result) in runs.items():
        print(f"{title:28}{len(shortest_path(result)):>7}{result.expanded:>12}"
              f"{result.generated:>11}{result.pruned:>9}")

    for name, (title, result) in runs.items():
        print(f"\n{title} path:", ", ".join(shortest_path(result)))
        on_path = highlight_path(result)
        result.tree.save2file(f"search_tree_{name}.txt", sorting=False)
        if name == "ordered":
            print("\nOrdered tree (depth <= 2, plus shortest path marked with **):")
            result.tree.show(
                filter=lambda node: node.data <= 2 or node.identifier in on_path,
                sorting=False,
            )
    print("\nFull trees written to search_tree_bfs.txt and search_tree_ordered.txt")
