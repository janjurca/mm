import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import os


def create_mask_from_jsonl(jsonl_line, img_height, img_width):
    # Create a black image
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Load JSON
    data = json.loads(jsonl_line)

    # Normalize the coordinates
    def normalize_coordinates(coord):
        center = img_width / 2, img_height / 2
        x = coord[0] + center[0]
        y = coord[1] + center[1]
        return [int(x), int(y)]

    # Process annotations
    for annotation in data['annotations']:
        # Normalize polygon coordinates
        for polygon in annotation['polygon']:
            pol = [normalize_coordinates(point) for point in polygon]
            contour = np.array([pol], dtype=np.int32)
            cv2.fillPoly(mask, contour, 255)

    return mask


parser = argparse.ArgumentParser(description='Create masks from jsonl files')
parser.add_argument('--dataset', type=str, help='Path to dataset folder')
args = parser.parse_args()

os.makedirs(args.dataset + '/masks', exist_ok=True)

with open(args.dataset + '/all_endoscopy.jsonl') as f:
    jsonl_lines = f.readlines()
    for jsonl_line in tqdm(jsonl_lines):
        data = json.loads(jsonl_line)
        image = cv2.imread(f"{args.dataset}/all_endoscopy/{data['media_id']}_{data['frame']}.jpg")
        img_height, img_width, _ = image.shape
        mask = create_mask_from_jsonl(jsonl_line, img_height, img_width)
        cv2.imwrite(args.dataset + '/masks/' + f"{data['media_id']}_{data['frame']}.jpg", mask)
