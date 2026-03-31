import pandas as pd
import json

def convert_csv_to_json(csv_path, json_path):
    df = pd.read_excel(csv_path)
    df = df.fillna(0)

    glasses_db = {}
    
    for index, row in df.iterrows():
        glasses_db[str(index)] = {
            "file_name": str(row.get('Model File Name', f'glass_{index}.glb')),
            "name": str(row.get('Name', '')),
            "shape_id": int(row.get('Shape ID', 0)),
            "material_id": int(row.get('Material ID', 0)),
            "rim_id": int(row.get('Rim ID', 0)),
            "width": float(row.get('Width (cm)', 0)) / 20.0,
            "height": float(row.get('Height (cm)', 0)) / 10.0,
            "bridge_pos": float(row.get('Bridge Pos', 0)),
            "normalized_material": float(row.get('Material ID', 0)) / 2.0,
            "normalized_rim": float(row.get('Rim ID', 0)) / 2.0
        }

    with open(json_path, 'w') as f:
        json.dump(glasses_db, f, indent=4)

if __name__ == "__main__":
    convert_csv_to_json('Glasses Features.xlsx', 'glasses_database.json')