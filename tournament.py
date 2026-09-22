"""Headless 4-player games (same rules loop as halma_pygame_4players.py).

Run:  python tournament.py
"""
import random
import statistics
import time
from typing import Callable, Dict, List

from AI_Player_Team33 import AI_Player_Team33, DISTANCE, square_name
from halma import check_legal_move, check_win_condition, initial_pos, move, parse_position

MAXIMUM_MOVE_LIMIT = 100
Bot = Callable[[List[List[int]], int, bool], tuple]


def _legal_moves(board, player):
    return [(r * 5 + c, nr * 5 + nc)
            for r in range(5) for c in range(5) if board[r][c] == player
            for nr in range(5) for nc in range(5) if check_legal_move(board, (r, c), (nr, nc))]


def random_bot(board, player, visualize_tree):
    frm, to = random.choice(_legal_moves(board, player))
    return square_name(frm), square_name(to)


def greedy_bot(board, player, visualize_tree):
    """One-ply: take the move that shrinks our own distance the most (random among ties)."""
    p = player - 1
    mask = sum(1 << (r * 5 + c) for r in range(5) for c in range(5) if board[r][c] == player)
    scored = [(DISTANCE[p][mask ^ (1 << frm) ^ (1 << to)], random.random(), frm, to)
              for frm, to in _legal_moves(board, player)]
    _, _, frm, to = min(scored)
    return square_name(frm), square_name(to)


def play_game(bots: Dict[int, Bot], timings: Dict[int, List[float]] = None):
    board = [row[:] for row in initial_pos]
    current, move_count = 1, 0
    result = check_win_condition(board, move_count, MAXIMUM_MOVE_LIMIT, True)
    while result.status == "ongoing":
        start = time.perf_counter()
        try:
            old_ref, new_ref = bots[current]([row[:] for row in board], current, False)
            moved = move(board, parse_position(old_ref), parse_position(new_ref), current)
        except Exception as error:
            print(f"  player {current} error: {type(error).__name__}: {error}")
            moved = False
        if timings is not None:
            timings.setdefault(current, []).append(time.perf_counter() - start)
        move_count += 1
        result = check_win_condition(board, move_count, MAXIMUM_MOVE_LIMIT, True)
        if result.status == "ongoing":
            current = current % 4 + 1
    return result, move_count


def points(result, player):
    if player in result.winners:
        return 3
    return 1 if player in result.tied_players else 0


def match(name: str, hero: Bot, others: Bot, games_per_seat: int = 10, seed: int = 1):
    """Put `hero` in each seat in turn against three copies of `others`."""
    random.seed(seed)
    wins = draws = 0
    lengths, hero_ms = [], []
    for seat in range(1, 5):
        for _ in range(games_per_seat):
            bots = {p: (hero if p == seat else others) for p in range(1, 5)}
            timings: Dict[int, List[float]] = {}
            result, turns = play_game(bots, timings)
            wins += seat in result.winners
            draws += result.status == "move_limit" and seat in result.tied_players
            lengths.append(turns)
            hero_ms += [1000 * t for t in timings[seat]]
    total = 4 * games_per_seat
    print(f"{name}: hero won {wins}/{total}, drew {draws}, avg game {statistics.mean(lengths):.0f} turns, "
          f"hero move time mean {statistics.mean(hero_ms):.1f} ms / max {max(hero_ms):.0f} ms")


if __name__ == "__main__":
    match("maxn vs 3 random", AI_Player_Team33, random_bot)
    match("maxn vs 3 greedy", AI_Player_Team33, greedy_bot)
    match("maxn vs 3 maxn (self-play)", AI_Player_Team33, AI_Player_Team33, games_per_seat=5)
