"""
Automatic cell mask generation using Cellpose for red-channel images.

Scans the input folder for TIFF files of the red channel (suffix *_z000_ch01.tif),
applies a pre-trained Cellpose-SAM model to segment cell membranes, and saves
the resulting label masks as NumPy arrays (*_mask.npy) into the Object_masks folder.

Run this module before interactive analysis to create segmentation masks
for all images in the selected experiment folder.
"""

import sys
import os
from pathlib import Path
from cellpose import models, io
import numpy as np

def run_cellpose_on_red_channel(input_folder, output_folder, diameter=100, use_gpu=True):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Создаём модель Cellpose-SAM с предобученной моделью cpsam (по умолчанию)
    model = models.CellposeModel(pretrained_model='cpsam', gpu=use_gpu)

    # Ищем файлы красного канала с окончанием _z000_ch01.tif
    image_paths = list(input_folder.glob("*_z000_ch01.tif"))
    if not image_paths:
        print("Файлы красного канала не найдены.")
        return

    for img_path in image_paths:
        print(f"Обработка: {img_path.name}")
        img = io.imread(str(img_path))

        # Приводим изображение к формату (C, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3:
            # Если изображение (H, W, C), меняем оси
            if img.shape[2] <= 3:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = np.transpose(img[:, :, :3], (2, 0, 1))
        else:
            raise ValueError(f"Неожиданный формат изображения: {img.shape}")

        # Запускаем сегментацию (channels параметр не нужен)
        masks, flows, styles = model.eval(img, diameter=diameter)

        # Сохраняем маску в формате .npy с суффиксом _mask.npy
        mask_path = output_folder / (img_path.stem + "_mask.npy")
        np.save(mask_path, masks)
        print(f"Маска сохранена: {mask_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Введите путь к папке с изображениями: ").strip('"').strip("'")

    if not os.path.exists(folder_path):
        print("Папка не найдена!")
        sys.exit(1)

    output_dir = os.path.join(folder_path, "Object_masks")
    run_cellpose_on_red_channel(folder_path, output_dir, diameter=100, use_gpu=True)
    print("Сегментация завершена.")