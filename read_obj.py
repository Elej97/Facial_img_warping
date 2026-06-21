def get_obj_bbox(filepath):
    min_v = [float('inf')] * 3
    max_v = [float('-inf')] * 3
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                v = [float(parts[1]), float(parts[2]), float(parts[3])]
                for i in range(3):
                    if v[i] < min_v[i]: min_v[i] = v[i]
                    if v[i] > max_v[i]: max_v[i] = v[i]
    size = [max_v[i] - min_v[i] for i in range(3)]
    print(f"--- {filepath} ---")
    print(f"Size: {size}")
    print(f"Min: {min_v}, Max: {max_v}")

get_obj_bbox('frontend/public/models/ties/Bowtie_01.obj')
