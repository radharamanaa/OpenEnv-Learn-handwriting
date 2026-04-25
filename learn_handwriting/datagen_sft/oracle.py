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
    "F": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical stem"},
        {"type": "line", "points": [(0.15, 0.05), (1.0, 0.05)], "desc": "top horizontal"},
        {"type": "line", "points": [(0.15, 0.48), (0.72, 0.48)], "desc": "middle horizontal"},
    ],
    "H": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "left vertical"},
        {"type": "line", "points": [(0.85, 0.0), (0.85, 1.0)], "desc": "right vertical"},
        {"type": "line", "points": [(0.15, 0.5), (0.85, 0.5)], "desc": "crossbar"},
    ],
    "I": [
        {"type": "line", "points": [(0.18, 0.06), (0.82, 0.06)], "desc": "top serif"},
        {"type": "line", "points": [(0.5, 0.06), (0.5, 0.94)], "desc": "vertical stem"},
        {"type": "line", "points": [(0.18, 0.94), (0.82, 0.94)], "desc": "bottom serif"},
    ],
    "K": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical spine"},
        {"type": "line", "points": [(0.15, 0.52), (0.92, 0.08)], "desc": "upper diagonal"},
        {"type": "line", "points": [(0.15, 0.48), (0.92, 0.92)], "desc": "lower diagonal"},
    ],
    "M": [
        {"type": "line", "points": [(0.07, 0.0), (0.07, 1.0)], "desc": "left stem"},
        {"type": "line", "points": [(0.07, 0.0), (0.5, 0.76)], "desc": "left roof to valley"},
        {"type": "line", "points": [(0.5, 0.76), (0.93, 0.0)], "desc": "valley to right peak"},
        {"type": "line", "points": [(0.93, 0.0), (0.93, 1.0)], "desc": "right stem"},
        {"type": "line", "points": [(0.5, 0.76), (0.5, 0.98)], "desc": "valley stem fill"},
    ],
    "W": [
        {"type": "line", "points": [(0.0, 0.0), (0.2, 1.0)], "desc": "left outer leg"},
        {"type": "line", "points": [(0.2, 1.0), (0.5, 0.08)], "desc": "left valley to center peak"},
        {"type": "line", "points": [(0.5, 0.08), (0.8, 1.0)], "desc": "center peak to right valley"},
        {"type": "line", "points": [(0.8, 1.0), (1.0, 0.0)], "desc": "right outer leg"},
        {"type": "line", "points": [(0.5, 0.08), (0.5, 0.48)], "desc": "center peak downward fill"},
        {"type": "line", "points": [(0.2, 1.0), (0.35, 0.55)], "desc": "left valley interior"},
        {"type": "line", "points": [(0.8, 1.0), (0.65, 0.55)], "desc": "right valley interior"},
    ],
    "Y": [
        {"type": "line", "points": [(0.1, 0.0), (0.5, 0.48)], "desc": "left upper diagonal"},
        {"type": "line", "points": [(0.9, 0.0), (0.5, 0.48)], "desc": "right upper diagonal"},
        {"type": "line", "points": [(0.5, 0.48), (0.5, 1.0)], "desc": "vertical stem"},
    ],
    "B": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical spine"},
        {"type": "curve", "points": [(0.15, 0.05), (0.15, 0.5), (0.95, 0.25)], "desc": "upper lobe"},
        {"type": "curve", "points": [(0.15, 0.5), (0.15, 0.95), (0.95, 0.75)], "desc": "lower lobe"},
    ],
    "D": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical spine"},
        {"type": "curve", "points": [(0.15, 0.04), (0.15, 0.96), (0.93, 0.5)], "desc": "curved belly"},
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
    "J": [
        {"type": "line", "points": [(0.58, 0.06), (0.76, 0.06)], "desc": "top serif"},
        {"type": "line", "points": [(0.76, 0.06), (0.76, 0.68)], "desc": "stem right"},
        {"type": "line", "points": [(0.58, 0.06), (0.58, 0.68)], "desc": "stem left"},
        {"type": "curve", "points": [(0.76, 0.66), (0.36, 0.93), (0.03, 0.58)], "desc": "bottom hook"},
    ],
    "P": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical spine"},
        {"type": "line", "points": [(0.15, 0.06), (0.56, 0.06)], "desc": "top bar"},
        {"type": "line", "points": [(0.56, 0.06), (0.74, 0.22)], "desc": "outer upper slant"},
        {"type": "line", "points": [(0.74, 0.22), (0.74, 0.55)], "desc": "outer right side"},
        {"type": "line", "points": [(0.74, 0.55), (0.15, 0.55)], "desc": "bowl bottom below counter"},
    ],
    "Q": [
        {"type": "ellipse", "points": [(0.5, 0.5), (0.45, 0.45)], "desc": "main ellipse body"},
        {"type": "line", "points": [(0.6, 0.6), (0.95, 0.95)], "desc": "diagonal tail"},
    ],
    "R": [
        {"type": "line", "points": [(0.15, 0.0), (0.15, 1.0)], "desc": "vertical spine"},
        {"type": "line", "points": [(0.15, 0.06), (0.56, 0.06)], "desc": "top bar"},
        {"type": "line", "points": [(0.56, 0.06), (0.74, 0.22)], "desc": "outer upper slant"},
        {"type": "line", "points": [(0.74, 0.22), (0.74, 0.52)], "desc": "outer right side"},
        {"type": "line", "points": [(0.74, 0.52), (0.15, 0.52)], "desc": "bowl bottom below counter"},
        {"type": "line", "points": [(0.15, 0.58), (0.86, 0.94)], "desc": "diagonal leg"},
    ],
    "U": [
        {"type": "line", "points": [(0.12, 0.06), (0.12, 0.66)], "desc": "left upright"},
        {"type": "curve", "points": [(0.12, 0.64), (0.5, 0.99), (0.88, 0.64)], "desc": "bottom cup"},
        {"type": "line", "points": [(0.88, 0.66), (0.88, 0.06)], "desc": "right upright"},
        {"type": "line", "points": [(0.3, 0.76), (0.7, 0.76)], "desc": "cup base fill"},
    ],
}
