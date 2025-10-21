# minimax_agent.py
from math import inf
try:
    # your run_game.py imports this; if present we'll use it as a cutoff eval
    from evaluation import betterEvaluationFunction
    HAVE_HEURISTIC = True
except Exception:
    HAVE_HEURISTIC = False


class MinimaxAgent:
    """
    Classic Minimax for Tic-Tac-Toe.

    Assumptions about GameState (from your run_game.py):
      - state.to_move            -> 'X' or 'O'
      - state.get_legal_actions()-> iterable of action ints (0..8)
      - state.generate_successor(a) -> NEW GameState after applying action a
      - state.is_terminal()      -> bool
      - state.winner()           -> 'X', 'O', or None
    """

    def __init__(self):
        # used by play_human_vs_ai; not required for AI vs AI
        self.symbol = None

    # ---------------------- public API ----------------------

    def get_action(self, state, depth_limit=None):
        """
        Choose the optimal action for whoever is to move in `state`.
        If depth_limit is None, searches the full game tree.
        """
        player = state.to_move
        best_action = None

        if player == 'X':  # MAX
            best_val = -inf
            for a in state.get_legal_actions():
                v = self._min_value(state.generate_successor(a),
                                    depth_limit, depth=1)
                if v > best_val:
                    best_val, best_action = v, a
        else:              # MIN (player == 'O')
            best_val = inf
            for a in state.get_legal_actions():
                v = self._max_value(state.generate_successor(a),
                                    depth_limit, depth=1)
                if v < best_val:
                    best_val, best_action = v, a

        return best_action

    # ---------------------- minimax core ----------------------

    def _max_value(self, state, depth_limit, depth):
        if self._cutoff(state, depth_limit, depth):
            return self._evaluate(state, at_cutoff=(depth_limit is not None and not state.is_terminal()))
        v = -inf
        for a in state.get_legal_actions():
            v = max(v, self._min_value(state.generate_successor(a), depth_limit, depth+1))
        return v

    def _min_value(self, state, depth_limit, depth):
        if self._cutoff(state, depth_limit, depth):
            return self._evaluate(state, at_cutoff=(depth_limit is not None and not state.is_terminal()))
        v = inf
        for a in state.get_legal_actions():
            v = min(v, self._max_value(state.generate_successor(a), depth_limit, depth+1))
        return v

    # ---------------------- helpers ----------------------

    def _cutoff(self, state, depth_limit, depth):
        if state.is_terminal():
            return True
        if depth_limit is None:
            return False
        return depth >= depth_limit

    def _evaluate(self, state, at_cutoff=False):
        """
        Returns a value from X's perspective:
            +1 if X wins, -1 if O wins, 0 for draw.
        If we stopped early due to depth_limit (at_cutoff=True), use a heuristic if available,
        otherwise return 0 (neutral).
        """
        w = state.winner()
        if w == 'X':
            return 1
        if w == 'O':
            return -1
        if state.is_terminal():
            return 0


