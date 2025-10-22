# minimax_agent.py
from math import inf
try:
    from evaluation import betterEvaluationFunction
    HAVE_HEURISTIC = True
except Exception:
    HAVE_HEURISTIC = False

class MinimaxAgent:
    """
    Classic Minimax for Tic-Tac-Toe.

    GameState is expected to provide:
      - state.to_move                 -> 'X' or 'O'
      - state.get_legal_actions()     -> iterable of actions (0..8)
      - state.generate_successor(a)   -> NEW state after action a
      - state.is_terminal()           -> bool
      - state.winner()                -> 'X', 'O', or None
    """

    def __init__(self):
        self.symbol = None  # used only in human-vs-ai mode

    def get_action(self, state, depth_limit=None):
        """Return the optimal action for whoever is to move in `state`."""
        player = state.to_move
        best_action = None

        if player == 'X':  # MAX
            best_val = -inf
            for a in state.get_legal_actions():
                v = self._min_value(state.generate_successor(a), depth_limit, depth=1)
                if v > best_val:
                    best_val, best_action = v, a
        else:              # MIN (O)
            best_val = inf
            for a in state.get_legal_actions():
                v = self._max_value(state.generate_successor(a), depth_limit, depth=1)
                if v < best_val:
                    best_val, best_action = v, a
        return best_action

    # ---------- minimax recursion ----------

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

    # ---------- helpers ----------

    def _cutoff(self, state, depth_limit, depth):
        if state.is_terminal():
            return True
        if depth_limit is None:
            return False
        return depth >= depth_limit

    def _evaluate(self, state, at_cutoff=False):
        """Return value from X's perspective: +1 win, -1 loss, 0 draw."""
        w = state.winner()
        if w == 'X':
            return 1
        if w == 'O':
            return -1
        if state.is_terminal():
            return 0
        if at_cutoff and HAVE_HEURISTIC:
            try:
                return float(betterEvaluationFunction(state))
            except Exception:
                return 0.0
        return 0.0
