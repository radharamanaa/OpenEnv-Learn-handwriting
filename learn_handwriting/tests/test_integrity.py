"""
Tests for shape integrity disqualification in the environment.

Run from the project root:
    pytest tests/test_integrity.py -v
"""

import numpy as np
import pytest

from models import LearnHandwritingAction
from server.learn_handwriting_environment import LearnHandwritingEnvironment
from server.renderer import DISQUALIFICATION_MASKS


def _env_for_char(char: str) -> LearnHandwritingEnvironment:
    """Create an environment and force it to the given character."""
    # Find which pool the character belongs to
    from server.learn_handwriting_environment import TASK_CHARACTERS
    task = next(
        (t for t, chars in TASK_CHARACTERS.items() if char in chars),
        "easy",
    )
    env = LearnHandwritingEnvironment(task=task)
    env._char_pool = [char]
    obs = env.reset()
    assert obs.target_character == char, f"Expected {char}, got {obs.target_character}"
    return env


# ── Integrity triggers correctly ────────────────────────────────────────────

def test_flood_canvas_triggers_integrity_A():
    """Filling the entire canvas must trigger A's triangle-interior constraint."""
    env = _env_for_char("A")
    env._canvas = np.ones((100, 100), dtype=np.int32)
    obs = env.step(LearnHandwritingAction(action_type="line", x1=0, y1=0, x2=1, y2=1))
    assert obs.done is True
    assert obs.integrity_violated is True
    assert obs.reward == 0.0, "Reward must be 0.0 on integrity violation (not negative)"


def test_flood_canvas_triggers_integrity_O():
    """Filling the entire canvas must trigger O's circle-interior constraint."""
    env = _env_for_char("O")
    env._canvas = np.ones((100, 100), dtype=np.int32)
    obs = env.step(LearnHandwritingAction(action_type="line", x1=0, y1=0, x2=1, y2=1))
    assert obs.done is True
    assert obs.integrity_violated is True
    assert obs.reward == 0.0


def test_flood_canvas_triggers_integrity_C():
    """Filling the entire canvas must trigger C's right-arc constraint (closing into O)."""
    env = _env_for_char("C")
    env._canvas = np.ones((100, 100), dtype=np.int32)
    obs = env.step(LearnHandwritingAction(action_type="line", x1=0, y1=0, x2=1, y2=1))
    assert obs.done is True
    assert obs.integrity_violated is True
    assert obs.reward == 0.0


def test_flood_canvas_triggers_integrity_B():
    """Filling the entire canvas must trigger B's lobe constraints."""
    env = _env_for_char("B")
    env._canvas = np.ones((100, 100), dtype=np.int32)
    obs = env.step(LearnHandwritingAction(action_type="line", x1=0, y1=0, x2=1, y2=1))
    assert obs.done is True
    assert obs.integrity_violated is True
    assert obs.reward == 0.0


# ── Clean strokes do NOT trigger integrity ───────────────────────────────────

def test_clean_left_diagonal_does_not_trigger_A():
    """The left diagonal of A must not fill the inner triangle."""
    env = _env_for_char("A")
    obs = env.step(LearnHandwritingAction(action_type="line", x1=10, y1=90, x2=50, y2=10))
    assert obs.integrity_violated is False
    assert obs.done is False or obs.match_percentage >= 0.90


def test_circle_for_O_does_not_trigger_integrity():
    """A well-placed circle stroke for O should NOT violate O's interior constraint."""
    env = _env_for_char("O")
    obs = env.step(LearnHandwritingAction(action_type="circle", x1=50, y1=50, radius=38))
    assert obs.integrity_violated is False


def test_vertical_spine_for_B_does_not_trigger_integrity():
    """The vertical spine stroke for B should not cover the lobe interiors."""
    env = _env_for_char("B")
    obs = env.step(LearnHandwritingAction(action_type="line", x1=20, y1=10, x2=20, y2=90))
    assert obs.integrity_violated is False


# ── Reward is never negative ─────────────────────────────────────────────────

def test_integrity_reward_is_not_negative():
    """Integrity violation must produce reward=0.0, never a negative value."""
    for char in ["A", "B", "C", "O", "G"]:
        if char not in DISQUALIFICATION_MASKS:
            continue
        env = _env_for_char(char)
        env._canvas = np.ones((100, 100), dtype=np.int32)
        obs = env.step(LearnHandwritingAction(action_type="line", x1=0, y1=0, x2=1, y2=1))
        assert obs.reward >= 0.0, f"{char}: reward was {obs.reward}, must be ≥ 0.0"


# ── Simple characters have no integrity constraints ──────────────────────────

def test_flood_canvas_triggers_integrity_Q():
    """Filling the entire canvas must trigger Q's circle-interior constraint."""
    env = _env_for_char("Q")
    env._canvas = np.ones((100, 100), dtype=np.int32)
    obs = env.step(LearnHandwritingAction(action_type="line", x1=0, y1=0, x2=1, y2=1))
    assert obs.done is True
    assert obs.integrity_violated is True
    assert obs.reward == 0.0


def test_circle_for_Q_does_not_trigger_integrity():
    """A well-placed circle stroke for Q should NOT violate Q's interior constraint."""
    env = _env_for_char("Q")
    obs = env.step(LearnHandwritingAction(action_type="circle", x1=50, y1=50, radius=38))
    assert obs.integrity_violated is False


def test_no_integrity_for_simple_chars():
    """Easy/medium chars without holes must never trigger integrity violation."""
    for char in ["L", "T", "V", "X", "N", "Z", "E"]:
        env = _env_for_char(char)
        env._canvas = np.ones((100, 100), dtype=np.int32)
        obs = env.step(LearnHandwritingAction(action_type="line", x1=0, y1=0, x2=99, y2=99))
        assert obs.integrity_violated is False, (
            f"{char}: unexpectedly triggered integrity_violated"
        )
