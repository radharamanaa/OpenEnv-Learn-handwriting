import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.learn_handwriting_environment import LearnHandwritingEnvironment, _draw_action
from models import LearnHandwritingAction

def test_l():
    env = LearnHandwritingEnvironment(task="easy")
    env._char_pool = ["L"]
    obs = env.reset(task="easy")
    target = env._target_matrix
    bbox = (obs.char_bbox_x1, obs.char_bbox_y1, obs.char_bbox_x2, obs.char_bbox_y2)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    best_match = 0
    best_points = None
    
    # Grid search for L
    for xv in np.linspace(0.05, 0.25, 5):
        for yh in np.linspace(0.85, 0.95, 5):
            for ytop in np.linspace(0.0, 0.1, 3):
                for xright in np.linspace(0.9, 1.0, 3):
                    canvas = np.zeros((100, 100), dtype=np.int32)
                    
                    x_v = int(bbox[0] + xv * w)
                    y_h = int(bbox[1] + yh * h)
                    y_t = int(bbox[1] + ytop * h)
                    x_r = int(bbox[0] + xright * w)
                    
                    x_v1 = int(bbox[0] + (xv - 0.05) * w)
                    x_v2 = int(bbox[0] + (xv + 0.05) * w)
                    y_h1 = int(bbox[1] + (yh - 0.05) * h)
                    y_h2 = int(bbox[1] + (yh + 0.05) * h)
                    
                    action1 = LearnHandwritingAction(action_type="line", x1=x_v1, y1=y_t, x2=x_v1, y2=y_h)
                    action2 = LearnHandwritingAction(action_type="line", x1=x_v2, y1=y_t, x2=x_v2, y2=y_h)
                    action3 = LearnHandwritingAction(action_type="line", x1=x_v, y1=y_h1, x2=x_r, y2=y_h1)
                    action4 = LearnHandwritingAction(action_type="line", x1=x_v, y1=y_h2, x2=x_r, y2=y_h2)
                    
                    m1 = _draw_action(action1)
                    m2 = _draw_action(action2)
                    m3 = _draw_action(action3)
                    m4 = _draw_action(action4)
                    canvas = np.maximum.reduce([m1, m2, m3, m4])
                    
                    match = np.sum(canvas * target) / np.sum(target)
                    if match > best_match:
                        best_match = match
                        best_points = (xv, yh, ytop, xright)
                        
    print(f"Best match for L: {best_match:.3f} with params: {best_points}")
    
if __name__ == "__main__":
    test_l()
