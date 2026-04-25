import itertools
import random
from typing import List, Dict, Any, Tuple

from datagen_sft.oracle import ORACLE_GEOMETRIES

def reverse_stroke(stroke: Dict[str, Any]) -> Dict[str, Any]:
    """Reverses the drawing direction of a stroke if applicable."""
    new_stroke = stroke.copy()
    points = stroke["points"].copy()
    
    if stroke["type"] == "line":
        # swap start and end
        points[0], points[1] = points[1], points[0]
        new_stroke["desc"] = new_stroke["desc"] + " (reversed)"
    elif stroke["type"] == "curve":
        # swap start and end, keep passthrough
        points[0], points[1] = points[1], points[0]
        new_stroke["desc"] = new_stroke["desc"] + " (reversed)"
    # circle/ellipse have no direction
    
    new_stroke["points"] = points
    return new_stroke

def generate_topological_variations(char: str) -> List[List[Dict[str, Any]]]:
    """Generate all combinations of stroke order and drawing directions for a character."""
    base_strokes = ORACLE_GEOMETRIES.get(char, [])
    if not base_strokes:
        return []
        
    variations = []
    # 1. Permute stroke order
    for ordered_strokes in itertools.permutations(base_strokes):
        # 2. For each stroke, 2 possible directions (except ellipse/circle)
        directions = []
        for stroke in ordered_strokes:
            if stroke["type"] in ["line", "curve"]:
                directions.append([stroke, reverse_stroke(stroke)])
            else:
                directions.append([stroke])
                
        # Cartesian product of directions
        for combination in itertools.product(*directions):
            variations.append(list(combination))
            
    return variations

def scale_and_jitter(
    stroke: Dict[str, Any], 
    bbox: Tuple[int, int, int, int], 
    jitter_range: int = 1
) -> List[Dict[str, Any]]:
    """Map normalized [0,1] coordinates to absolute bounding box coordinates with jitter.
    Returns a LIST of offset parallel strokes to simulate a thicker brush and achieve 90% coverage."""
    bx1, by1, bx2, by2 = bbox
    w = bx2 - bx1
    h = by2 - by1
    
    # We will return 2 parallel strokes, offset horizontally to widen vertical/diagonal lines
    # and offset vertically to widen horizontal lines. Since we don't calculate normals,
    # we'll just offset diagonally which covers both somewhat:
    offsets = [(-0.03, 0.03), (0.03, -0.03)]
    
    results = []
    
    for ox, oy in offsets:
        new_stroke = stroke.copy()
        new_points = []
        for i, pt in enumerate(stroke["points"]):
            if stroke["type"] == "ellipse" and i == 1:
                rx = (pt[0] + abs(ox)) * w
                ry = (pt[1] + abs(oy)) * h
                rx = max(1, int(rx + random.randint(-jitter_range, jitter_range)))
                ry = max(1, int(ry + random.randint(-jitter_range, jitter_range)))
                new_points.append((rx, ry))
            elif stroke["type"] == "circle" and i == 1:
                r = (pt[0] + abs(ox)) * max(w, h)
                r = max(1, int(r + random.randint(-jitter_range, jitter_range)))
                new_points.append((r,))
            else:
                nx, ny = pt
                ax = bx1 + (nx + ox) * w
                ay = by1 + (ny + oy) * h
                
                ax += random.randint(-jitter_range, jitter_range)
                ay += random.randint(-jitter_range, jitter_range)
                
                ax = max(0, min(99, int(ax)))
                ay = max(0, min(99, int(ay)))
                new_points.append((ax, ay))
                
        new_stroke["points"] = new_points
        new_stroke["desc"] = new_stroke.get("desc", "") + f" (offset {ox},{oy})"
        results.append(new_stroke)
        
    return results

def get_augmented_episodes(char: str, bbox: Tuple[int, int, int, int], num_augments: int = 2) -> List[List[Dict[str, Any]]]:
    """Returns a list of augmented episodes (each episode is a list of strokes)."""
    base_variations = generate_topological_variations(char)
    all_episodes = []
    
    if len(base_variations) > 50:
        base_variations = random.sample(base_variations, 50)
        
    for variation in base_variations:
        for _ in range(num_augments):
            augmented_episode = []
            for stroke in variation:
                aug_strokes = scale_and_jitter(stroke, bbox, jitter_range=2)
                augmented_episode.extend(aug_strokes)
            all_episodes.append(augmented_episode)
            
    return all_episodes
