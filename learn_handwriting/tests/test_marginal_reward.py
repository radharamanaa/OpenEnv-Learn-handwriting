"""Reward must reflect new coverage only, not re-traced target pixels."""

from learn_handwriting import LearnHandwritingAction
from server.learn_handwriting_environment import LearnHandwritingEnvironment


def test_redundant_stroke_gives_zero_reward_and_zero_matched_this_step():
    env = LearnHandwritingEnvironment(task="easy")
    obs = env.reset(task="easy", character="L")
    assert obs.match_percentage == 0.0

    # First stroke: horizontal bar of L (adjust if needed — use a line that hits target)
    a1 = LearnHandwritingAction(action_type="line", x1=20, y1=25, x2=80, y2=25)
    o1 = env.step(a1)
    assert o1.pixels_matched_this_stroke > 0
    assert o1.reward > 0
    cov1 = o1.match_percentage

    # Repeat identical stroke: no new target pixels
    o2 = env.step(a1)
    assert o2.pixels_matched_this_stroke == 0
    assert o2.reward == 0.0
    assert o2.match_percentage == cov1
