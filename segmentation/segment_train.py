import shutil
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import List
import torch
import json
import numpy as np
import cv2
from skimage.io import imread, imsave
import argparse
import random
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support
from lightning.pytorch.loggers import MLFlowLogger
import os
import torchvision.transforms as T
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import mlflow
import matplotlib

os.environ["MLFLOW_TRACKING_USERNAME"] = "username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"


class SegmentationDataset(Dataset):
    def __init__(
        self,
        path_to_data: str,
        items: List[dict],
    ) -> None:
        self.path_to_data = path_to_data
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        image_path = f"{self.path_to_data}/all_endoscopy/{item['media_id']}_{item['frame']}.jpg"

        image = imread(image_path)
        image_width, image_height = image.shape[1], image.shape[0]
        mask, polygon = self.create_mask_from_jsonl(item, image_height, image_width)
        rotation_center = self.polygon_center(polygon)

        if False:
            plt.subplot(2, 2, 1)  # 1 row, 2 columns, plot position 1
            plt.imshow(image)
            plt.subplot(2, 2, 2)  # 1 row, 2 columns, plot position 2
            plt.imshow(mask)
            plt.scatter(int(rotation_center[0]), int(rotation_center[1]), c='red', s=50, marker='o')

        augs = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Affine(translate_px={"x": int((image_width/2 - rotation_center[0])), "y": int(image_height/2 - rotation_center[1])}),
            iaa.Affine(rotate=(-25, 25),),
            iaa.Fliplr(0.5),  # horizontal flips
        ])  # apply augmenters in random order

        mask = np.expand_dims(mask, axis=2)
        mask = np.expand_dims(mask, axis=0)

        keypoints = [Keypoint(x=p[0], y=p[1]) for p in polygon]
        kps = KeypointsOnImage(keypoints, shape=image.shape)
        image, mask, polygon = augs(image=image, segmentation_maps=mask, keypoints=kps)
        mask = mask.squeeze(0)

        if False:
            plt.subplot(2, 2, 3)  # 1 row, 2 columns, plot position 1
            plt.imshow(image)
            plt.subplot(2, 2, 4)  # 1 row, 2 columns, plot position 2
            plt.imshow(mask)
            plt.show()

        bounding_box = self.extract_bounding_box(polygon.to_xy_array())

        image = self.crop_transformed_image(image, polygon.to_xy_array())
        mask = self.crop_transformed_image(mask, polygon.to_xy_array())
        scaler = T.Resize((224, 224), antialias=True)  # height and width as a tuple
        image = torch.tensor(image.transpose(2, 0, 1))
        mask = torch.tensor(mask.transpose(2, 0, 1))
        try:
            image = scaler(image)
            mask = scaler(mask)
        except RuntimeError as e:
            print(image_path, rotation_center, bounding_box, image.shape, mask.shape)
            raise e

        result = {
            "image": image,
            "mask": mask / 255.0,
            "filename": image_path,
        }

        return result

    def crop_transformed_image(self, original_image, transformed_points, offset=5):
        min_x, min_y, max_x, max_y = self.extract_bounding_box(transformed_points)
        min_x = min_x - offset
        min_y = min_y - offset
        max_x = max_x + offset
        max_y = max_y + offset

        # Clip the bounding box to ensure it remains within the image bounds
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, original_image.shape[1])
        max_y = min(max_y, original_image.shape[0])

        # Crop the image using the adjusted bounding box coordinates
        cropped_image = original_image[int(min_y):int(max_y), int(min_x):int(max_x)]
        return cropped_image

    def extract_bounding_box(self, transformed_points):
        min_x = np.min(transformed_points[:, 0])
        max_x = np.max(transformed_points[:, 0])
        min_y = np.min(transformed_points[:, 1])
        max_y = np.max(transformed_points[:, 1])

        # Return the bounding box as (x_min, y_min, x_max, y_max)
        return min_x, min_y, max_x, max_y

    @staticmethod
    def create_mask_from_jsonl(data, img_height, img_width):
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        def normalize_coordinates(coord):
            center = img_width / 2, img_height / 2
            x = coord[0] + center[0]
            y = coord[1] + center[1]
            return [int(x), int(y)]

        for annotation in data['annotations']:
            # Normalize polygon coordinates
            polygon = random.choice(annotation['polygon'])
            pol = [normalize_coordinates(point) for point in polygon]
            contour = np.array([pol], dtype=np.int32)
            cv2.fillPoly(mask, contour, 255)

        return mask, pol

    def polygon_center(self, points_list):
        points_array = np.array(points_list)
        min_x = np.min(points_array[:, 0])
        max_x = np.max(points_array[:, 0])
        min_y = np.min(points_array[:, 1])
        max_y = np.max(points_array[:, 1])

        x_center = (min_x + (max_x-min_x)/2)
        y_center = (min_y + (max_y-min_y)/2)
        return x_center, y_center


class SegmentationModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.metrics = {"train": [], "valid": [], "test": []}

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, stage):
        outputs = self.metrics[stage]
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_loss": torch.stack([x["loss"] for x in outputs]).mean(),
        }
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def training_step(self, batch, batch_idx):
        step = self.shared_step(batch, "train")
        self.metrics["train"].append(step)
        return step

    def on_training_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        step = self.shared_step(batch, "valid")
        self.metrics["valid"].append(step)
        return step

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        step = self.shared_step(batch, "test")
        self.metrics["test"].append(step)
        return step

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.metrics["valid"] = []

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.metrics["train"] = []

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.metrics["test"] = []


def inference(model, image_path, saveto=None):
    image = imread(image_path)
    model.eval()
    model.cpu()
    os.makedirs(saveto, exist_ok=True)
    with torch.no_grad():
        image = torch.tensor(image.transpose(2, 0, 1))
        scaler = T.Resize((224, 224), antialias=True)  # height and width as a tuple
        image = scaler(image)
        image = torch.unsqueeze(image, 0)

        logit_mask = model.forward(image)
        prob_mask = logit_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        mask = pred_mask.squeeze().cpu().numpy()

        mask_img = Image.fromarray(mask, "L")
        mask_img.save(f"{saveto}/{image_path.split('/')[-1].split('.')[0]}_output_mask.jpg")

        plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot position 1
        plt.imshow(image.squeeze().cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot position 2
        plt.imshow(mask)
        plt.show()


def load_data(manifest):
    data = []
    with open(manifest) as f:
        for line in f:
            data.append(json.loads(line))
    media_ids = [x["media_id"] for x in data]

    train_media_ids, valid_media_ids = train_test_split(data, test_size=0.2)
    valid_media_ids, test_media_ids = train_test_split(valid_media_ids, test_size=0.5)

    train_data = [x for x in data if x["media_id"] in train_media_ids]
    valid_data = [x for x in data if x["media_id"] in valid_media_ids]
    test_data = [x for x in data if x["media_id"] in test_media_ids]

    return train_data, valid_data, test_data


if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--dataset', type=str, help='Path to dataset folder')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_cpu', type=int, default=1, help='Number of CPU threads')
    parser.add_argument('--accelerator', type=str, default='cpu', help='Accelerator')
    parser.add_argument('--save_to', type=str, default='results', help='saveto')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint')
    parser.add_argument('--inference', action='store', help='Inference image path')
    parser.add_argument('--arch', type=str, default='FPN', help='Architecture')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = SegmentationModel(args.arch, args.encoder, in_channels=3, out_classes=1)

    if args.checkpoint is not None:
        model = model.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))
    if args.inference is not None:
        inference(model, args.inference, args.save_to, )
        exit()

    train, valid, test = load_data(os.path.join(args.dataset, "manifest.json"))

    train_dataloader = DataLoader(SegmentationDataset(args.dataset, train), batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    valid_dataloader = DataLoader(SegmentationDataset(args.dataset, valid), batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
    test_dataloader = DataLoader(SegmentationDataset(args.dataset, test), batch_size=1, shuffle=False, num_workers=args.n_cpu)

    mlf_logger = MLFlowLogger(experiment_name="endoscopy_segmentation", tracking_uri="url", log_model=False,
                              tags={"mlflow.runName": f"{args.arch}_{args.encoder}", "model": args.arch, "encoder": args.encoder, "in_channels": "3", "out_classes": "1"})
    mlf_logger.log_hyperparams({"batch_size": args.batch_size, "seed": args.seed, "n_cpu": args.n_cpu, "accelerator": args.accelerator})

    trainer = pl.Trainer(accelerator=args.accelerator, logger=mlf_logger, max_epochs=args.epochs, default_root_dir="checkpoints/")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    os.makedirs(args.save_to, exist_ok=True)

    for i, item in enumerate(test_dataloader):
        if i > 32:
            break
        logit_mask = model.forward(item["image"])
        prob_mask = logit_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        for i, filename in enumerate(item["filename"]):
            print(filename)
            print(pred_mask[i].shape)
            mask = pred_mask[i].squeeze(0).cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask, "L")
            mask_img.save(f"{args.save_to}/{filename.split('/')[-1].split('.')[0]}_output_mask.jpg")

            img = item["image"][i].cpu().numpy().transpose(1, 2, 0)
            imsave(f"{args.save_to}/{filename.split('/')[-1].split('.')[0]}_img.jpg", img)

            gt_mask = item["mask"][i].cpu().numpy().squeeze(0)
            gt_mask = (gt_mask * 255).astype(np.uint8)
            gt_mask = Image.fromarray(gt_mask, "L")
            gt_mask.save(f"{args.save_to}/{filename.split('/')[-1].split('.')[0]}_gt.jpg")
