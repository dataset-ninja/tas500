import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_data_path = "/home/alex/DATASETS/TODO/TAS500/tas500v1.1/train"
    val_data_path = "/home/alex/DATASETS/TODO/TAS500/tas500v1.1/val"
    test_data_path = "/home/alex/DATASETS/TODO/TAS500/tas500v1.1/test"
    train_masks_path = "/home/alex/DATASETS/TODO/TAS500/tas500v1.1/train_labels_ids"
    val_masks_path = "/home/alex/DATASETS/TODO/TAS500/tas500v1.1/val_labels_ids"

    batch_size = 30

    ds_name_to_split = {
        "train": (train_data_path, train_masks_path),
        "val": (val_data_path, val_masks_path),
        "test": (test_data_path, None),
    }

    def create_ann(image_path):
        labels = []

        img_height = 620
        img_wight = 2026

        mask_path = os.path.join(masks_path, get_file_name_with_ext(image_path))
        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
        unique_pixels = np.unique(mask_np)

        for pixel in unique_pixels:
            tags = []
            supercategory_value = pixel_to_supercategory.get(pixel)
            if supercategory_value is not None:
                supercategory = sly.Tag(supercategory_meta, value=supercategory_value)
                tags.append(supercategory)

            obj_class = pixel_to_class.get(pixel)

            mask = mask_np == pixel
            curr_bitmap = sly.Bitmap(mask)
            curr_label = sly.Label(curr_bitmap, obj_class, tags=tags)
            labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    supercategory_meta = sly.TagMeta("supercategory", sly.TagValueType.ANY_STRING)

    pixel_to_class = {
        0: sly.ObjClass("asphalt", sly.Bitmap, color=(192, 192, 192)),
        1: sly.ObjClass("gravel", sly.Bitmap, color=(105, 105, 105)),
        2: sly.ObjClass("soil", sly.Bitmap, color=(160, 82, 45)),
        3: sly.ObjClass("sand", sly.Bitmap, color=(244, 164, 96)),
        4: sly.ObjClass("bush", sly.Bitmap, color=(60, 179, 113)),
        5: sly.ObjClass("forest", sly.Bitmap, color=(34, 139, 34)),
        6: sly.ObjClass("low grass", sly.Bitmap, color=(154, 205, 50)),
        7: sly.ObjClass("high grass", sly.Bitmap, color=(0, 128, 0)),
        8: sly.ObjClass("misc. vegetation", sly.Bitmap, color=(0, 100, 0)),
        9: sly.ObjClass("tree crown", sly.Bitmap, color=(0, 250, 154)),
        10: sly.ObjClass("tree trunk", sly.Bitmap, color=(139, 69, 19)),
        11: sly.ObjClass("building", sly.Bitmap, color=(1, 51, 73)),
        12: sly.ObjClass("fence", sly.Bitmap, color=(190, 153, 153)),
        13: sly.ObjClass("wall", sly.Bitmap, color=(0, 132, 111)),
        14: sly.ObjClass("car", sly.Bitmap, color=(0, 0, 142)),
        15: sly.ObjClass("bus", sly.Bitmap, color=(0, 60, 100)),
        16: sly.ObjClass("sky", sly.Bitmap, color=(135, 206, 250)),
        17: sly.ObjClass("misc. object", sly.Bitmap, color=(128, 0, 128)),
        18: sly.ObjClass("pole", sly.Bitmap, color=(153, 153, 153)),
        19: sly.ObjClass("traffic sign", sly.Bitmap, color=(255, 255, 0)),
        20: sly.ObjClass("person", sly.Bitmap, color=(220, 20, 60)),
        21: sly.ObjClass("animal", sly.Bitmap, color=(255, 182, 193)),
        22: sly.ObjClass("ego vehicle", sly.Bitmap, color=(220, 220, 220)),
        255: sly.ObjClass("undefined", sly.Bitmap, color=(0, 0, 0)),
    }

    pixel_to_supercategory = {
        0: "terrain",
        1: "terrain",
        2: "terrain",
        3: "terrain",
        4: "vegetation",
        5: "vegetation",
        6: "vegetation",
        7: "vegetation",
        8: "vegetation",
        9: "vegetation",
        10: "vegetation",
        11: "construction",
        12: "construction",
        13: "construction",
        14: "vehicle",
        15: "vehicle",
        16: "sky",
        17: "object",
        18: "object",
        19: "object",
        20: "human",
        21: "animal",
        22: "void",
    }

    meta = sly.ProjectMeta(
        tag_metas=[supercategory_meta], obj_classes=list(pixel_to_class.values())
    )

    api.project.update_meta(project.id, meta.to_json())

    for ds_name, data_path in ds_name_to_split.items():
        dataset = api.dataset.create(project.id, ds_name)

        images_path, masks_path = data_path

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_path) for image_path in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            if masks_path is not None:
                anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
                api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
