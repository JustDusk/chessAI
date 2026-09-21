"""Team 33 Halma player: maxn search over the 4-player 5x5 board.

Entry point: AI_Player_Team33(board, player, visualize_tree) -> ("A1", "B1").
"""
import itertools
import operator
import os
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from treelib import Tree

TEAM_NUMBER = 33
MY_MOVES_AHEAD = 2   # how many of OUR OWN moves the search looks ahead

SIZE = 5
NUM_SQUARES = SIZE * SIZE
NUM_PLAYERS = 4
PIECES_PER_PLAYER = 3
WIN_SCORE = 1000
MISSING_PIECE_PENALTY = 10   # keeps a player with <3 pieces from ever "winning"

Vector = Tuple[int, int, int, int]      # one score per player (index = player - 1)
Move = Tuple[int, int, int, int]        # (ordering key, from square, to square, piece index)

# End zones per player as (row, col); same cells as halma.win_cells_all.
_GOAL_CELLS = [
    [(3, 4), (4, 3), (4, 4)],   # player 1: E4 D5 E5
    [(3, 0), (4, 0), (4, 1)],   # player 2: A4 A5 B5
    [(0, 0), (0, 1), (1, 0)],   # player 3: A1 B1 A2
    [(0, 3), (0, 4), (1, 4)],   # player 4: D1 E1 E2
]
_FAR_CORNER = [(4, 4), (4, 0), (0, 0), (0, 4)]   # deepest cell of each end zone


def square_name(square: int) -> str:
    return chr(ord("A") + square % SIZE) + str(square // SIZE + 1)


def _build_move_table() -> List[List[Tuple[int, int]]]:
    """For every square: (target, jumped-over square or -1 for a plain step)."""
    table: List[List[Tuple[int, int]]] = []
    for square in range(NUM_SQUARES):
        row, col = divmod(square, SIZE)
        options = []
        for d_row, d_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            for length in (1, 2):
                new_row, new_col = row + d_row * length, col + d_col * length
                if 0 <= new_row < SIZE and 0 <= new_col < SIZE:
                    middle = -1
                    if length == 2:
                        middle = (row + d_row) * SIZE + (col + d_col)
                    options.append((new_row * SIZE + new_col, middle))
        table.append(options)
    return table


MOVES_FROM = _build_move_table()
GOAL_SQUARES = [[r * SIZE + c for r, c in cells] for cells in _GOAL_CELLS]
# PROGRESS[p][sq]: higher = closer to player p's far corner (cheap ordering hint)
PROGRESS = [
    [-(abs(sq // SIZE - corner[0]) + abs(sq % SIZE - corner[1])) for sq in range(NUM_SQUARES)]
    for corner in _FAR_CORNER
]


class _DistanceTable(dict):
    """mask of a player's pieces -> distance still to travel (0 = all pieces home).

    Lazily filled: __missing__ runs once per distinct piece set, then it is a plain dict lookup.
    """

    def __init__(self, to_goal: Sequence[Sequence[int]]) -> None:
        """to_goal[g][sq] = cost for a single piece to get from square sq to goal cell g."""
        super().__init__()
        self.to_goal = to_goal

    def __missing__(self, mask: int) -> int:
        squares = [sq for sq in range(NUM_SQUARES) if (mask >> sq) & 1]
        best = min(
            sum(self.to_goal[g][sq] for sq, g in zip(squares, assignment))
            for assignment in itertools.permutations(range(PIECES_PER_PLAYER), len(squares))
        )
        value = best + MISSING_PIECE_PENALTY * (PIECES_PER_PLAYER - len(squares))
        self[mask] = value
        return value


def manhattan_tables(player: int) -> "_DistanceTable":
    goals = [divmod(sq, SIZE) for sq in GOAL_SQUARES[player]]
    return _DistanceTable([
        [abs(sq // SIZE - r) + abs(sq % SIZE - c) for sq in range(NUM_SQUARES)] for r, c in goals
    ])


DISTANCE = [manhattan_tables(p) for p in range(NUM_PLAYERS)]
WIN_VECTORS: List[Vector] = [
    tuple(WIN_SCORE if k == p else -WIN_SCORE for k in range(NUM_PLAYERS))  # type: ignore[misc]
    for p in range(NUM_PLAYERS)
]


class Entry(NamedTuple):
    value: Vector
    best_move: Optional[Move]
    searched: int      # children actually explored (fewer than `total` means pruned)
    total: int


class MaxNSearch:
    """maxn search from `player` (0-based) on a flat 25-cell board (0 empty, 1-4 pieces)."""

    def __init__(self, cells: Sequence[int], player: int, my_moves: int = MY_MOVES_AHEAD,
                 ordering: bool = True, pruning: bool = True, use_tt: bool = True) -> None:
        self.board = list(cells)
        self.pieces = [[sq for sq in range(NUM_SQUARES) if self.board[sq] == p + 1]
                       for p in range(NUM_PLAYERS)]
        self.mask = [sum(1 << sq for sq in squares) for squares in self.pieces]
        self.key = sum(mask << (NUM_SQUARES * p) for p, mask in enumerate(self.mask))
        self.dist = [DISTANCE[p][self.mask[p]] for p in range(NUM_PLAYERS)]

        self.root_player = player
        self.plies = NUM_PLAYERS * (my_moves - 1) + 1
        self.mover = {level: (player + self.plies - level) % NUM_PLAYERS
                      for level in range(1, self.plies + 1)}
        self.ordering, self.pruning, self.use_tt = ordering, pruning, use_tt
        self.tt: List[Dict[int, Entry]] = [{} for _ in range(self.plies + 1)]
        self.nodes = 0
        self.tt_hits = 0
        self.root: Entry

    # ---- moves -------------------------------------------------------------
    def ordered_moves(self, player: int) -> List[Move]:
        board = self.board
        progress = PROGRESS[player]
        moves: List[Move] = []
        for index, frm in enumerate(self.pieces[player]):
            for to, middle in MOVES_FROM[frm]:
                if board[to] == 0 and (middle < 0 or board[middle] != 0):
                    moves.append((progress[frm] - progress[to], frm, to, index))
        if self.ordering:
            moves.sort()   # most negative key first = biggest step toward the far corner
        return moves

    def _make(self, player: int, frm: int, to: int, index: int) -> int:
        self.board[frm] = 0
        self.board[to] = player + 1
        self.pieces[player][index] = to
        delta = (1 << frm) | (1 << to)
        self.mask[player] ^= delta
        self.key ^= delta << (NUM_SQUARES * player)
        old = self.dist[player]
        self.dist[player] = DISTANCE[player][self.mask[player]]
        return old

    def _unmake(self, player: int, frm: int, to: int, index: int, old_dist: int) -> None:
        self.board[to] = 0
        self.board[frm] = player + 1
        self.pieces[player][index] = frm
        delta = (1 << frm) | (1 << to)
        self.mask[player] ^= delta
        self.key ^= delta << (NUM_SQUARES * player)
        self.dist[player] = old_dist

    # ---- search ------------------------------------------------------------
    def _leaf(self) -> Vector:
        d = self.dist
        return (-d[0], -d[1], -d[2], -d[3])

    def _after_move(self, level: int, player: int) -> Vector:
        if self.dist[player] == 0:
            return WIN_VECTORS[player]
        if level == 1:
            return self._leaf()
        return self._search(level - 1)

    def _ceiling(self, player: int, level: int) -> int:
        """Best own score `player` could still reach: every remaining move gains at most 2."""
        own_moves = (level + 3) // 4
        distance = self.dist[player]
        return WIN_SCORE if distance <= 2 * own_moves else 2 * own_moves - distance

    def _search(self, level: int) -> Vector:
        key = self.key
        if self.use_tt:
            entry = self.tt[level].get(key)
            if entry is not None:
                self.tt_hits += 1
                return entry.value
        self.nodes += 1

        player = self.mover[level]
        moves = self.ordered_moves(player)
        best_move: Optional[Move] = None
        searched = 0
        if not moves:                      # blocked player passes
            best = self._after_move(level, player) if level > 1 else self._leaf()
        else:
            ceiling = self._ceiling(player, level) if self.pruning else None
            best = None
            for move in moves:
                searched += 1
                _, frm, to, index = move
                old = self._make(player, frm, to, index)
                value = self._after_move(level, player)
                self._unmake(player, frm, to, index, old)
                if best is None or value[player] > best[player]:
                    best, best_move = value, move
                    if ceiling is not None and value[player] >= ceiling:
                        break              # nothing left can beat this
        entry = Entry(best, best_move, searched, len(moves))
        if self.use_tt:
            self.tt[level][key] = entry
        if level == self.plies:
            self.root = entry
        return best

    def best_move(self) -> Optional[Tuple[int, int]]:
        """(from square, to square) chosen for the root player, or None if it cannot move."""
        self._search(self.plies)
        move = self.root.best_move
        return None if move is None else (move[1], move[2])

    def best_move_text(self) -> str:
        move = self.best_move()
        return "none" if move is None else f"{square_name(move[0])}->{square_name(move[1])}"


# ---- public interface -------------------------------------------------------
def _validate_input(board, player, visualize_tree) -> Tuple[List[int], int, bool]:
    """Check the three arguments; return (flat cells, 0-based player, visualise flag)."""
    try:
        rows = list(board)
    except TypeError:
        raise TypeError("board must be a 5x5 grid of integers") from None
    if len(rows) != SIZE:
        raise ValueError(f"board must have {SIZE} rows, got {len(rows)}")

    cells: List[int] = []
    for r, row in enumerate(rows):
        try:
            row = list(row)
        except TypeError:
            raise TypeError(f"board row {r + 1} is not a sequence") from None
        if len(row) != SIZE:
            raise ValueError(f"board row {r + 1} must have {SIZE} cells, got {len(row)}")
        for c, cell in enumerate(row):
            where = chr(ord("A") + c) + str(r + 1)
            if isinstance(cell, bool):
                raise TypeError(f"cell {where} must be an integer 0-4, got {cell!r}")
            try:
                value = operator.index(cell)
            except TypeError:
                raise TypeError(f"cell {where} must be an integer 0-4, got {cell!r}") from None
            if not 0 <= value <= NUM_PLAYERS:
                raise ValueError(f"cell {where} must be between 0 and 4, got {value}")
            cells.append(value)

    if isinstance(player, bool):
        raise TypeError("player must be an integer from 1 to 4")
    try:
        player = operator.index(player)
    except TypeError:
        raise TypeError(f"player must be an integer from 1 to 4, got {player!r}") from None
    if not 1 <= player <= NUM_PLAYERS:
        raise ValueError(f"player must be between 1 and 4, got {player}")

    for colour in range(1, NUM_PLAYERS + 1):
        if cells.count(colour) > PIECES_PER_PLAYER:
            raise ValueError(f"player {colour} has more than {PIECES_PER_PLAYER} pieces on the board")
    if cells.count(player) == 0:
        raise ValueError(f"player {player} has no pieces on the board")

    if visualize_tree not in (0, 1):
        raise ValueError(f"visualize_tree must be True/False or 0/1, got {visualize_tree!r}")
    return cells, player - 1, bool(visualize_tree)


def AI_Player_Team33(board, player, visualize_tree=False) -> Tuple[str, str]:
    """Return (from, to) such as ("A1", "C1") for `player` on `board`."""
    cells, player0, visualize = _validate_input(board, player, visualize_tree)
    search = MaxNSearch(cells, player0)
    move = search.best_move()
    if visualize:
        save_tree_image(search)
    if move is None:   # completely blocked: an illegal move makes the game skip our turn
        own = square_name(search.pieces[player0][0])
        return own, own
    return square_name(move[0]), square_name(move[1])


# ---- search tree visualisation ---------------------------------------------
def _vector_text(value: Vector) -> str:
    for player, win in enumerate(WIN_VECTORS):
        if value == win:
            return f"P{player + 1} WINS"
    return "[" + " ".join(f"{score:>3}" for score in value) + "]"


def build_display_tree(search: MaxNSearch) -> Tree:
    """treelib tree of what the search did: all first replies, plus the best line in full.

    Every node is "P<n>: from->to  [scores of P1..P4]". The best line is wrapped in **...**.
    Children the pruning never looked at are marked (pruned).
    """
    tree = Tree()
    counter = itertools.count()
    root = search.root
    tree.create_node(
        f"P{search.root_player + 1} to move, best {search.best_move_text()}  {_vector_text(root.value)}",
        next(counter))

    def expand(parent: int, level: int, on_best_line: bool) -> None:
        player = search.mover[level]
        entry = search.tt[level][search.key]
        moves = search.ordered_moves(player)
        if not moves:
            node = tree.create_node(f"P{player + 1}: no legal move (pass)", next(counter), parent=parent)
            if level > 1:
                expand(node.identifier, level - 1, on_best_line)
            return
        for position, move in enumerate(moves):
            _, frm, to, index = move
            label = f"P{player + 1}: {square_name(frm)}->{square_name(to)}"
            if position >= entry.searched:
                tree.create_node(f"{label}  (pruned)", next(counter), parent=parent)
                continue
            old = search._make(player, frm, to, index)
            value = search._after_move(level, player)
            is_best = on_best_line and move == entry.best_move
            text = f"{label}  {_vector_text(value)}"
            node = tree.create_node(f"**{text}**" if is_best else text, next(counter), parent=parent)
            can_go_deeper = level > 1 and search.dist[player] != 0
            if can_go_deeper and (is_best or level == search.plies):
                expand(node.identifier, level - 1, is_best)
            search._unmake(player, frm, to, index, old)

    expand(0, search.plies, True)
    return tree


def save_tree_image(search: MaxNSearch, path: Optional[str] = None) -> str:
    """Write the display tree to Team<X>_Tree.png (text file if pygame is unavailable)."""
    path = path or f"Team{TEAM_NUMBER}_Tree.png"
    pruned = sum(e.total - e.searched for level in search.tt for e in level.values())
    header = [
        f"Team {TEAM_NUMBER} maxn search tree: player {search.root_player + 1} to move, "
        f"{search.plies} plies (2 own moves)",
        f"nodes searched {search.nodes}, repeated positions reused {search.tt_hits}, "
        f"children pruned {pruned}",
        "Scores are [P1 P2 P3 P4] = minus the distance each player still has to travel (higher is better).",
        "Each player picks the child with the best score for THEMSELVES. The chosen best line is highlighted.",
        "",
    ]
    lines = build_display_tree(search).show(stdout=False, sorting=False, line_type="ascii-ex").splitlines()
    try:
        import pygame
    except ImportError:
        path = os.path.splitext(path)[0] + ".txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(header + lines))
        return path

    pygame.font.init()
    names = "consolas,dejavusansmono,couriernew,monospace"
    font, bold = pygame.font.SysFont(names, 15), pygame.font.SysFont(names, 15, bold=True)
    height = font.get_linesize()
    pad = 12
    width = max(font.size(text)[0] for text in header + lines) + 2 * pad
    surface = pygame.Surface((width, height * (len(header) + len(lines)) + 2 * pad))
    surface.fill((255, 255, 255))
    y = pad
    for text in header:
        surface.blit(font.render(text, True, (60, 60, 60)), (pad, y))
        y += height
    for text in lines:
        if "**" in text:
            pygame.draw.rect(surface, (255, 243, 191), (0, y, width, height))
            surface.blit(bold.render(text.replace("**", ""), True, (170, 30, 40)), (pad, y))
        elif "(pruned)" in text:
            surface.blit(font.render(text, True, (150, 150, 150)), (pad, y))
        else:
            surface.blit(font.render(text, True, (25, 30, 45)), (pad, y))
        y += height
    pygame.image.save(surface, path)
    return path
