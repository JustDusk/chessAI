"""Sanity tests for AI_Player_Team33. Run: python test_maxn.py"""
import os
import random

from AI_Player_Team33 import AI_Player_Team33, MaxNSearch, square_name
from halma import check_legal_move, initial_pos, move


def random_positions(count, seed=0):
    """Positions reached by random legal play from the starting board."""
    rng = random.Random(seed)
    positions = []
    while len(positions) < count:
        board = [row[:] for row in initial_pos]
        for turn in range(rng.randint(0, 60)):
            player = turn % 4 + 1
            moves = [((r, c), (nr, nc)) for r in range(5) for c in range(5) if board[r][c] == player
                     for nr in range(5) for nc in range(5) if check_legal_move(board, (r, c), (nr, nc))]
            if moves:
                move(board, *rng.choice(moves), player)
        positions.append(board)
    return positions


def flat(board):
    return [cell for row in board for cell in row]


def test_move_generation_matches_halma(positions):
    for board in positions:
        for player in range(1, 5):
            expected = {(r * 5 + c, nr * 5 + nc) for r in range(5) for c in range(5) if board[r][c] == player
                        for nr in range(5) for nc in range(5) if check_legal_move(board, (r, c), (nr, nc))}
            search = MaxNSearch(flat(board), player - 1)
            got = {(frm, to) for _, frm, to, _ in search.ordered_moves(player - 1)}
            assert got == expected, (board, player, got ^ expected)
    print(f"move generation matches halma.check_legal_move on {len(positions)} positions x 4 players: OK")


def test_pruning_never_changes_the_answer(positions):
    """Pruning + transposition table must return exactly what full maxn returns."""
    checked = 0
    for board in positions[:120]:
        for player in range(4):
            full = MaxNSearch(flat(board), player, ordering=True, pruning=False, use_tt=False)
            fast = MaxNSearch(flat(board), player, ordering=True, pruning=True, use_tt=True)
            assert full.best_move() == fast.best_move(), (board, player)
            assert full.root.value == fast.root.value, (board, player)
            checked += 1
    print(f"pruned search == full maxn (same move, same value vector) on {checked} root positions: OK")


def test_input_validation():
    bad_inputs = [
        ([[0] * 5] * 4, 1, 0),                       # 4 rows
        (initial_pos, 5, 0),                         # player out of range
        (initial_pos, True, 0),                      # player is a bool
        (initial_pos, 1, 7),                         # bad flag
        ([[9, 0, 0, 0, 0]] + [[0] * 5] * 4, 1, 0),   # cell value out of range
        ([[1.5] * 5] * 5, 1, 0),                     # non-integer cells
        ("hello", 1, 0),                             # not a board
        ([[0] * 5] * 5, 1, 0),                       # player has no pieces
        ([[1] * 5] * 5, 1, 0),                       # too many pieces
    ]
    for args in bad_inputs:
        try:
            AI_Player_Team33(*args)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"accepted invalid input {args!r}")
    print(f"input validation rejects {len(bad_inputs)} kinds of bad input: OK")


def test_blocked_player_and_tree_output():
    # Player 1 on A1 B1 A2; pieces of players 2-4 sit on every square it could step or jump to.
    board = [[1, 1, 2, 2, 0],
             [1, 3, 2, 0, 0],
             [3, 3, 0, 0, 0],
             [4, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
    search = MaxNSearch([c for row in board for c in row], 0)
    assert search.ordered_moves(0) == [], "test board is not actually blocked"
    assert AI_Player_Team33(board, 1, False) == ("A1", "A1")
    assert AI_Player_Team33(board, 1, True) == ("A1", "A1")
    assert os.path.exists("Team33_Tree.png")
    print("fully blocked player returns a harmless move and its tree is still drawn: OK")


if __name__ == "__main__":
    positions = random_positions(300)
    test_move_generation_matches_halma(positions)
    test_pruning_never_changes_the_answer(positions)
    test_input_validation()
    test_blocked_player_and_tree_output()
