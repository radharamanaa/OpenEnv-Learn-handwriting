# datagen_sft/oracle.py

from typing import Dict, List, Any

# The Geometric Oracle
# Geometries are defined in normalized [0, 1] coordinates relative to the character's bounding box.
# Points for line: [(x1,y1), (x2,y2)]
# Points for curve: [(x1,y1), (x2,y2), (x3,y3)]  (x3,y3 is the pass-through point)
# Points for ellipse: [(cx,cy), (rx_norm, ry_norm)]
# Points for circle: [(cx,cy), (r_norm)]

ORACLE_GEOMETRIES: Dict[str, List[Dict[str, Any]]] = {
    "L": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical stem"},
        {"type": "line", "points": [(0.15, 0.95), (1.0, 0.95)], "desc": "horizontal base"},
    ],
    "T": [
        {"type": "line", "points": [(0.0, 0.05), (1.0, 0.05)], "desc": "horizontal head"},
        {"type": "line", "points": [(0.5, 0.05), (0.5, 1.0)], "desc": "vertical stem"},
    ],
    "V": [
        {"type": "line", "points": [(0.0, 0.0), (0.5, 1.0)], "desc": "left diagonal"},
        {"type": "line", "points": [(0.5, 1.0), (1.0, 0.0)], "desc": "right diagonal"},
    ],
    "X": [
        {"type": "line", "points": [(0.0, 0.0), (1.0, 1.0)], "desc": "forward diagonal"},
        {"type": "line", "points": [(1.0, 0.0), (0.0, 1.0)], "desc": "backward diagonal"},
    ],
    "A": [
        {"type": "line", "points": [(0.5, 0.0), (0.1, 1.0)], "desc": "left diagonal"},
        {"type": "line", "points": [(0.5, 0.0), (0.9, 1.0)], "desc": "right diagonal"},
        {"type": "line", "points": [(0.25, 0.6), (0.75, 0.6)], "desc": "horizontal crossbar"},
    ],
    "N": [
        {"type": "line", "points": [(0.15, 1.0), (0.15, 0.0)], "desc": "left vertical"},
        {"type": "line", "points": [(0.15, 0.0), (0.85, 1.0)], "desc": "diagonal connector"},
        {"type": "line", "points": [(0.85, 1.0), (0.85, 0.0)], "desc": "right vertical"},
    ],
    "Z": [
        {"type": "line", "points": [(0.0, 0.05), (1.0, 0.05)], "desc": "top horizontal"},
        {"type": "line", "points": [(1.0, 0.05), (0.0, 0.95)], "desc": "diagonal"},
        {"type": "line", "points": [(0.0, 0.95), (1.0, 0.95)], "desc": "bottom horizontal"},
    ],
    "E": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical back"},
        {"type": "line", "points": [(0.15, 0.05), (1.0, 0.05)], "desc": "top horizontal"},
        {"type": "line", "points": [(0.15, 0.5), (0.8, 0.5)], "desc": "middle horizontal"},
        {"type": "line", "points": [(0.15, 0.95), (1.0, 0.95)], "desc": "bottom horizontal"},
    ],
    "B": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical spine"},
        {"type": "curve", "points": [(0.15, 0.05), (0.15, 0.5), (0.95, 0.25)], "desc": "upper lobe"},
        {"type": "curve", "points": [(0.15, 0.5), (0.15, 0.95), (0.95, 0.75)], "desc": "lower lobe"},
    ],
    "C": [
        {"type": "curve", "points": [(0.9, 0.1), (0.9, 0.9), (0.05, 0.5)], "desc": "main C-curve"},
    ],
    "S": [
        {"type": "curve", "points": [(0.9, 0.1), (0.5, 0.5), (0.1, 0.25)], "desc": "top loop"},
        {"type": "curve", "points": [(0.5, 0.5), (0.1, 0.9), (0.9, 0.75)], "desc": "bottom loop"},
    ],
    "O": [
        {"type": "ellipse", "points": [(0.5, 0.5), (0.45, 0.45)], "desc": "outer ellipse loop"},
    ],
    "G": [
        {"type": "curve", "points": [(0.9, 0.1), (0.5, 0.95), (0.05, 0.5)], "desc": "main C-shape curve"},
        {"type": "line", "points": [(0.5, 0.95), (0.9, 0.95)], "desc": "bottom hook connector"},
        {"type": "line", "points": [(0.9, 0.95), (0.9, 0.5)], "desc": "hook vertical drop"},
        {"type": "line", "points": [(0.9, 0.5), (0.6, 0.5)], "desc": "hook horizontal cross"},
    ],
    "Q": [
        {"type": "ellipse", "points": [(0.5, 0.5), (0.45, 0.45)], "desc": "main ellipse body"},
        {"type": "line", "points": [(0.6, 0.6), (0.95, 0.95)], "desc": "diagonal tail"},
    ],
}
