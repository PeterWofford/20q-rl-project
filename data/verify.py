import json
import math

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_best_attribute(candidates, available_attributes):
    """
    Finds the attribute that splits the candidates most evenly (closest to 50/50).
    This minimizes the depth of the tree (greedy approximation of optimal).
    """
    best_attr = None
    min_diff = float('inf')
    
    total_candidates = len(candidates)
    
    for attr in available_attributes:
        # Count how many objects have this attribute
        true_count = sum(1 for obj in candidates if obj['attrs'].get(attr, False))
        false_count = total_candidates - true_count
        
        # We want to minimize the difference between the two groups
        diff = abs(true_count - false_count)
        
        # If this split is better, save it
        if diff < min_diff:
            min_diff = diff
            best_attr = attr
            
    return best_attr

def build_optimal_tree(candidates, available_attributes, current_depth=0, results=None):
    """
    Recursively builds a decision tree to find the depth for each object.
    """
    if results is None:
        results = {}

    # BASE CASE 1: Solution Found
    if len(candidates) == 1:
        obj = candidates[0]
        results[obj['name']] = {
            'depth': current_depth,
            'id': obj['id']
        }
        return results

    # BASE CASE 2: No attributes left (Indistinguishable objects)
    if not available_attributes and len(candidates) > 1:
        names = [c['name'] for c in candidates]
        print(f"WARNING: Indistinguishable objects found: {names}")
        for c in candidates:
            results[c['name']] = {
                'depth': current_depth, 
                'id': c['id'],
                'note': 'Ambiguous'
            }
        return results

    # RECURSIVE STEP: Find best question
    best_attr = get_best_attribute(candidates, available_attributes)
    
    # If no attribute can split the remaining candidates (all have same values for remaining attrs)
    if best_attr is None:
         for c in candidates:
            results[c['name']] = {'depth': current_depth, 'id': c['id'], 'note': 'Ambiguous'}
         return results

    # Split the group
    true_group = [c for c in candidates if c['attrs'].get(best_attr, False)]
    false_group = [c for c in candidates if not c['attrs'].get(best_attr, False)]
    
    # Remove the used attribute so we don't ask it again
    next_attributes = [a for a in available_attributes if a != best_attr]
    
    if true_group:
        build_optimal_tree(true_group, next_attributes, current_depth + 1, results)
    if false_group:
        build_optimal_tree(false_group, next_attributes, current_depth + 1, results)
        
    return results

# --- EXECUTION ---
objects = load_data('objects.json')

# Get all unique attributes from the first object (assuming schema is consistent)
# Or union all keys if sparse
all_attributes = list(objects[0]['attrs'].keys())

print(f" Analyzing {len(objects)} objects with {len(all_attributes)} attributes...")

# Run the calculator
depths = build_optimal_tree(objects, all_attributes)

# --- REPORTING ---
print(f"{'OBJECT NAME':<20} | {'OPTIMAL Qs':<10}")
print("-" * 35)

total_depth = 0
max_depth = 0
min_depth = float('inf')

sorted_objects = sorted(depths.items(), key=lambda x: x[1]['depth'])

for name, data in sorted_objects:
    d = data['depth']
    total_depth += d
    max_depth = max(max_depth, d)
    min_depth = min(min_depth, d)
    print(f"{name:<20} | {d:<10}")

avg_depth = total_depth / len(objects)

print("-" * 35)
print(f"Total Objects: {len(objects)}")
print(f"Average Questions: {avg_depth:.2f}")
print(f"Minimum Questions: {min_depth}")
print(f"Maximum Questions: {max_depth}")
print(f"Theoretical Limit (log2): {math.log2(len(objects)):.2f}")