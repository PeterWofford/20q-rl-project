import json

def fix_data():
    with open('objects.json', 'r') as f: objects = json.load(f)
    with open('attributes.json', 'r') as f: attributes = json.load(f)

    # 1. Add new attributes
    new_attrs = ["is_building", "has_drawers"]
    for attr in new_attrs:
        if attr not in attributes: attributes.append(attr)

    # 2. Update Objects
    for obj in objects:
        name = obj['name']
        attrs = obj['attrs']
        
        # Init new attributes
        for attr in new_attrs:
            if attr not in attrs: attrs[attr] = False

        # Apply Logical Fixes
        if name in ["Hospital", "School", "Library"]:
            attrs["is_building"] = True
        
        if name == "School":
            attrs["makes_sound"] = True
            
        if name in ["Desk", "Cabinet"]:
            attrs["has_drawers"] = True

    # Save
    with open('objects.json', 'w') as f: json.dump(objects, f, indent=2)
    with open('attributes.json', 'w') as f: json.dump(attributes, f, indent=2)
    print("Dataset patched successfully.")

fix_data()