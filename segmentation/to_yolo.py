import os
import json
import argparse
from PIL import Image


def convert_to_yolo_format(jsonl_file):
    root_dir = os.path.dirname(jsonl_file)
    splitname = os.path.basename(jsonl_file).split('.')[0]
    print(splitname)
    print(root_dir)
    print(jsonl_file)
    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            image_path = f"{root_dir}/{splitname}/{entry['media_id']}_{entry['frame']}.jpg"

            # Check if the image exists and open it
            img = Image.open(image_path)
            width, height = img.size

            yolo_lines = []

            def normalize_coordinates(coord):
                center = width / 2, height / 2
                x = coord[0] + center[0]
                y = coord[1] + center[1]
                return [int(x)/width, int(y)/height]

            # Process annotations
            for annotation in entry['annotations']:
                # Normalize polygon coordinates
                for polygon in annotation['polygon']:
                    pol = [normalize_coordinates(point) for point in polygon]

                    # For YOLO segmentation, we use the class ID followed by the normalized polygon
                    yolo_line = ['0'] + [f"{x[0]} {x[1]}" for x in pol]
                    yolo_lines.append(" ".join(yolo_line))

            # Write to a text file with the same name as the image but with .txt extension
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            with open(txt_path, 'w') as txt_file:
                txt_file.write("\n".join(yolo_lines))


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to YOLO segmentation format.")
    parser.add_argument('jsonl_file', help='Path to the JSONL file.')
    args = parser.parse_args()

    convert_to_yolo_format(args.jsonl_file)


if __name__ == "__main__":
    main()
