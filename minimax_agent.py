# minimax_agent.py
from math import inf
from agent_base import AgentBase
from evaluation import evaluation  

class MinimaxAgent(AgentBase):
    """
    Plays optimal Tic-Tac-Toe using classic Minimax.
    Assumes the game object exposes:
      - game.current_player (or game.player_to_move())  -> 'X' or 'O'
      - game.legal_moves()                              -> iterable of actions (e.g., ints 0..8)
      - game.is_terminal()                              -> bool
      - game.winner()                                   -> 'X', 'O', or None
      - game.play(action) or game.result(action)        -> returns a NEW game state after applying action
    If your names differ, adjust the calls in _max_value/_min_value accordingly.
    """

    def __init__(self, my_mark='X'):
        super().__init__(my_mark)

    def choose_action(self, game):
        """Return the optimal action for the current player."""
        assert hasattr(game, "legal_moves"), "game.legal_moves() missing"
        best_score = -inf if self._is_max(game) else inf
        best_action = None

        for a in game.legal_moves():
            child = self._next_state(game, a)

            if self._is_max(game):  # we are MAX at root if it's our turn as 'X' (or based on my_mark)
                score = self._min_value(child)
                if score > best_score:
                    best_score, best_action = score, a
            else:  # root is MIN (rare if agent plays 'O'); we still pick best for our perspective
                score = self._max_value(child)
                if score < best_score:
                    best_score, best_action = score, a

        return best_action

    def _max_value(self, game):
        if game.is_terminal():
            return self._terminal_score(game)

        v = -inf
        for a in game.legal_moves():
            v = max(v, self._min_value(self._next_state(game, a)))
        return v

    def _min_value(self, game):
        if game.is_terminal():
            return self._terminal_score(game)

        v = inf
        for a in game.legal_moves():
            v = min(v, self._max_value(self._next_state(game, a)))
        return v

    def _terminal_score(self, game):
        """Return +1 if my_mark wins, -1 if opponent wins, 0 for draw."""
        w = game.winner()  # 'X' or 'O' or None
        if w is None:
            return 0
        return 1 if w == self.my_mark else -1

    def _next_state(self, game, action):
        """Return a NEW game state after applying action (adapt to your API)."""
        if hasattr(game, "result"):
            return game.result(action)
        g2 = game.copy()
        g2.play(action)
        return g2

    def _is_max(self, game):
        """Decide whether the root node should maximize or minimize.

        Convention: 'X' is MAX, 'O' is MIN. If your framework defines
        otherwise, flip the check below.
        """
        player_to_move = getattr(game, "current_player", None)
        if player_to_move is None and hasattr(game, "player_to_move"):
            player_to_move = game.player_to_move()
        return (self.my_mark == 'X' and player_to_move == 'X') or \
               (self.my_mark == 'O' and player_to_move == 'X')  # keep MAX as 'X' by convention

