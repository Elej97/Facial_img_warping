import json
import struct
import sys

def print_glb_info(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'glTF':
            print(f"{filepath} is not a valid GLB")
            return
        version = struct.unpack('<I', f.read(4))[0]
        length = struct.unpack('<I', f.read(4))[0]
        chunk_len = struct.unpack('<I', f.read(4))[0]
        chunk_type = f.read(4)
        if chunk_type != b'JSON':
            print("First chunk is not JSON")
            return
        
        json_data = f.read(chunk_len).decode('utf-8')
        gltf = json.loads(json_data)
        
        print(f"--- {filepath} ---")
        if 'meshes' in gltf:
            for i, mesh in enumerate(gltf['meshes']):
                print(f"Mesh {i}: {mesh.get('name', 'unnamed')}")
        
        if 'nodes' in gltf:
            for i, node in enumerate(gltf['nodes']):
                if 'scale' in node:
                    print(f"Node {i} scale: {node['scale']}")
                if 'translation' in node:
                    print(f"Node {i} trans: {node['translation']}")
                    
        # Find min/max in accessors
        if 'accessors' in gltf:
            for i, acc in enumerate(gltf['accessors']):
                if acc.get('type') == 'VEC3' and 'min' in acc and 'max' in acc:
                    min_v = acc['min']
                    max_v = acc['max']
                    size = [max_v[j] - min_v[j] for j in range(3)]
                    print(f"Accessor {i} (VEC3) size: {size} (min: {min_v}, max: {max_v})")

print_glb_info('frontend/public/models/ties/Ties.glb')
print_glb_info('frontend/public/models/ties/Necktie.glb')
