"""
Interactive cell selection and intensity measurement for Avtocellseek.

Loads red and green channel images together with Cellpose masks, shows an overlay
for each image, and lets the user select individual cells by mouse clicks.
For selected cells, the script calculates background-corrected red and green
intensities, computes green/red ratios, and saves per-series and combined CSV
tables suitable for further statistical analysis and curve fitting.
"""
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage import io
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.stats import shapiro, ttest_ind
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

green_black_cmap = LinearSegmentedColormap.from_list("green_black", [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
red_black_cmap = LinearSegmentedColormap.from_list("red_black", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

plt.ion()

class CellSelector:
    """Интерактивный выбор клеток с подсветкой и подсчётом интенсивностей."""

    def __init__(self, red_img, green_img, cellpose_mask, img_name,
             total_selected_cells=0, series=None, ring_thickness=3):
        self.series = series
        self.red_img = red_img
        self.green_img = green_img
        self.cellpose_mask = cellpose_mask
        self.img_name = img_name
        self.selected_cells = set()
        self.total_selected_cells = total_selected_cells
        self.finished = False
        self.ring_thickness = ring_thickness
        self.results = []
        
        # ------ база для оверлея по красному каналу ----------
        self.img_for_overlay = gray2rgb(self.red_img) if self.red_img.ndim == 2 else self.red_img
        base_img = self.img_for_overlay.astype(np.float32)
        m = base_img.max()
        if m > 0:
            base_img /= m
        
        color_map = matplotlib.colormaps['tab10']
        excluded_index = 7
        valid_indices = [i for i in range(10) if i != excluded_index]
        
        unique_cells = np.unique(self.cellpose_mask)
        unique_cells = unique_cells[unique_cells != 0]
        
        for idx, cell_id in enumerate(unique_cells):
            cell_mask = (self.cellpose_mask == cell_id)
            color_idx = valid_indices[idx % len(valid_indices)]
            color = np.array(color_map(color_idx)[:3])
            alpha = 0.2
            base_img[cell_mask] = (1 - alpha) * base_img[cell_mask] + alpha * color
        
        self.overlay_base = base_img
        self.overlay_current = self.overlay_base.copy()
        
        # ---------- фигура и оси ----------
        self.fig, (self.ax_green, self.ax_overlay, self.ax_red) = plt.subplots(
            1, 3, figsize=(18, 7), dpi=80
        )
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15, wspace=0.05)
    
        # зелёный канал слева
        self.ax_green.imshow(self.green_img, cmap=green_black_cmap, vmin=0, vmax=255)
        self.ax_green.set_title('Зеленый канал')
        self.ax_green.axis('off')
    
        # центральный оверлей
        self.im = self.ax_overlay.imshow(self.overlay_current, vmin=0.0, vmax=1.0)
        self.title_text = self.ax_overlay.set_title(self._title(), fontsize=12)
        self.ax_overlay.axis('off')
    
        # красный канал справа
        self.ax_red.imshow(self.red_img, cmap=red_black_cmap, vmin=0, vmax=255)
        self.ax_red.set_title('Красный канал')
        self.ax_red.axis('off')
    
        # одинаковые пределы осей
        for ax in (self.ax_green, self.ax_overlay, self.ax_red):
            ax.set_xlim(0, self.cellpose_mask.shape[1])
            ax.set_ylim(self.cellpose_mask.shape[0], 0)
    
        # кнопка завершения
        ax_button = plt.axes([0.4, 0.03, 0.2, 0.07])
        self.btn = Button(ax_button, 'Завершить выбор')
        self.btn.on_clicked(self.finish_selection)
    
        # обработчик кликов только по центральной оси
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager.window, 'state'):
                manager.window.state('zoomed')
        except Exception:
            pass

    def _title(self):
        return f"Изображение: {self.img_name} | Выбрано: {len(self.selected_cells)} | Всего в серии: {self.total_selected_cells}"

    def highlight_selected_cells(self):
        overlay_img = self.overlay_base.copy()
        red_fill = np.array([1.0, 0.0, 0.0])
        alpha_red = 0.1
    
        for cell_id in self.selected_cells:
            cell_mask = (self.cellpose_mask == cell_id)
            if not np.any(cell_mask):
                continue
    
            eroded_mask = binary_erosion(cell_mask, iterations=self.ring_thickness)
            inner_ring = cell_mask & (~eroded_mask)
            if np.sum(inner_ring) == 0:
                inner_ring = cell_mask
    
            overlay_img[inner_ring] = (1 - alpha_red) * overlay_img[inner_ring] + alpha_red * red_fill
    
            boundaries = find_boundaries(inner_ring, mode='outer')
            boundaries_thick = binary_dilation(boundaries, iterations=1)
            overlay_img[boundaries_thick] = red_fill
    
        self.overlay_current = overlay_img
        self.im.set_data(self.overlay_current)
        self.title_text.set_text(self._title())
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        if self.finished or event.inaxes != self.ax_overlay:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < self.cellpose_mask.shape[1] and 0 <= y < self.cellpose_mask.shape[0]):
            return
        cell_id = self.cellpose_mask[y, x]
        if cell_id == 0:
            print("Кликнули по фону, клетка не выбрана.")
            return

        if cell_id in self.selected_cells:
            print(f"Снимаем выделение с клетки {cell_id}")
            self.selected_cells.remove(cell_id)
        else:
            print(f"Выбрана клетка {cell_id}")
            self.selected_cells.add(cell_id)
        self.highlight_selected_cells()

    def finish_selection(self, event):
        print("Завершение выбора клеток.")
        self.finished = True
        plt.close(self.fig)

    def calculate_intensities(self):
        all_cells_mask = (self.cellpose_mask > 0)
        background_mask = ~all_cells_mask

        background_red = np.mean(self.red_img[background_mask]) if np.any(background_mask) else 0
        background_green = np.mean(self.green_img[background_mask]) if np.any(background_mask) else 0

        rows = []
        for cell_id in self.selected_cells:
            cell_mask = (self.cellpose_mask == cell_id)
            if np.sum(cell_mask) == 0:
                continue

            eroded_mask = binary_erosion(cell_mask, iterations=3)
            inner_ring = cell_mask & (~eroded_mask)
            if np.sum(inner_ring) == 0:
                inner_ring = cell_mask

            if np.any(inner_ring):
                red_intensity_raw = np.mean(self.red_img[inner_ring])
                green_intensity_raw = np.mean(self.green_img[inner_ring])
                red_intensity = max(red_intensity_raw - background_red, 0)
                green_intensity = max(green_intensity_raw - background_green, 0)
            else:
                red_intensity_raw = green_intensity_raw = red_intensity = green_intensity = np.nan

            ratio = green_intensity / red_intensity if red_intensity > 0 else np.nan

            rows.append({
                'image': self.img_name,
                'cell_id': cell_id,
                'background_red': background_red,
                'background_green': background_green,
                'green_intensity_raw': green_intensity_raw,
                'red_intensity_raw': red_intensity_raw,
                'green_intensity': green_intensity,
                'red_intensity': red_intensity,
                'ratio': ratio,
                'series': self.series
            })
        self.results = rows

def load_red_green(folder_path, base_name):
    red_path = os.path.join(folder_path, f"{base_name}_ch01.tif")
    green_path = os.path.join(folder_path, f"{base_name}_ch00.tif")
    cellpose_path = os.path.join(folder_path, "Object_masks", f"{base_name}_ch01_mask.npy")

    red_img = io.imread(red_path)
    green_img = io.imread(green_path)
    cellpose_mask = np.load(cellpose_path, allow_pickle=True)
    if isinstance(cellpose_mask, dict) and 'masks' in cellpose_mask:
        cellpose_mask = cellpose_mask['masks']
    if not isinstance(cellpose_mask, np.ndarray) or cellpose_mask.ndim != 2:
        raise ValueError(f"Cellpose mask должен быть 2D numpy массивом, а получен {type(cellpose_mask)} с ndim={getattr(cellpose_mask, 'ndim', None)}")
    return red_img, green_img, cellpose_mask

def iterative_outlier_removal(data):
    data = np.array(data)
    cleaned = data.copy()
    p_val_best = 0
    improved = True

    while improved and len(cleaned) > 3:
        stat, p_val = shapiro(cleaned)
        if p_val > 0.05:
            break

        outlier_idx = np.argmax(np.abs(cleaned - np.mean(cleaned)))
        temp = np.delete(cleaned, outlier_idx)
        stat_new, p_val_new = shapiro(temp)

        if p_val_new > p_val_best and p_val_new > p_val:
            cleaned = temp
            p_val_best = p_val_new
        else:
            improved = False
    return cleaned

def statistical_analysis(df, sort_param, folder_path, series):
    """
    Проверка на нормальность по выбранному параметру (red_intensity_raw или ratio).
    df — DataFrame со всеми клетками после интерактивного выбора.
    sort_param — имя столбца для сортировки и фильтрации.
    background_value — значение фона (для визуализации).
    """
    data_raw = df[sort_param].dropna().values

    stat_before, p_before = shapiro(data_raw)
    cleaned_data = iterative_outlier_removal(data_raw)
    stat_after, p_after = shapiro(cleaned_data)

    plt.figure(figsize=(10, 6))
    plt.hist(data_raw, bins=30, alpha=0.5, label=f'До очистки\np={p_before:.3f}')
    plt.hist(cleaned_data, bins=30, alpha=0.5, label=f'После очистки\np={p_after:.3f}')
    plt.title(f'Проверка на нормальность по Шапиро-Уилку ({sort_param})')
    plt.xlabel(f'{sort_param}')
    plt.ylabel('Частота')
    plt.legend()
    plt.show()

    print()
    print(f"Шапиро-Уилка до очистки: p = {p_before:.4f}")
    print(f"Шапиро-Уилка после очистки: p = {p_after:.4f}\n")

    while True:
        user_input = input("Согласны ли вы с фильтрацией данных? (да/нет): ").strip().lower()
        if user_input == 'да':
            return cleaned_data, 'очищенная'
        elif user_input == 'нет':
            return data_raw, 'оригинальная'
        else:
            print("Некорректный ввод. Введите 'да' или 'нет'.")

def calculate_and_save_statistics(df, folder_path, series):
    stats = {}
    cols = ['background_green', 'background_red', 'green_intensity_raw', 'red_intensity_raw',
            'green_intensity', 'red_intensity', 'ratio']
    for col in cols:
        numeric_col = pd.to_numeric(df[col], errors='coerce')  # <- ЗДЕСЬ df вместо df_filtered
        valid = numeric_col.dropna()
        stats[f'{col}_mean'] = valid.mean()
        stats[f'{col}_std'] = valid.std()
        stats[f'{col}_sem'] = valid.sem()
        stats[f'{col}_count'] = valid.count()
    stats_df = pd.DataFrame([stats])
    stats_df['series'] = series
    
    stat_folder = os.path.join(folder_path, 'Statistics')
    os.makedirs(stat_folder, exist_ok=True)
    
    stats_df.to_csv(os.path.join(stat_folder, f'stats_series_{series}.csv'), index=False)

series_pattern = re.compile(r'_(\d+)_.*_z000_ch01')
def extract_series_number(filename):
    match = series_pattern.search(filename)
    if match:
        return match.group(1)
    return None
    
def main():
    if len(sys.argv) < 3:
        print("Ошибка: не указан путь к папке или номер серии.")
        print("Использование: python cell_analysis.py <путь_к_папке> <номер_серии> [имя_поля_для_сортировки]")
        return

    folder_path = sys.argv[1]
    series = sys.argv[2]

    # параметр сортировки: либо из argv, либо спросить у пользователя
    if len(sys.argv) >= 4:
        sort_param = sys.argv[3]
    else:
        sort_param = input("Введите параметр для сортировки (например, red_intensity_raw или ratio): ").strip()

    if not os.path.isdir(folder_path):
        print(f"Указанный путь не существует или не является папкой: {folder_path}")
        return

    files = os.listdir(folder_path)
    base_names = [Path(f).stem[:-5] for f in files if f.endswith('_ch01.tif') and extract_series_number(f) == series]
    if not base_names:
        print(f"Нет файлов для серии {series}")
        return

    total_selected_cells = 0
    all_results = []

    for base_name in sorted(base_names):
        print(f"Обработка изображения: {base_name}")
        try:
            red_img, green_img, cellpose_mask = load_red_green(folder_path, base_name)
        except Exception as e:
            print(f"Ошибка загрузки данных для {base_name}: {e}")
            continue
        selector = CellSelector(red_img, green_img, cellpose_mask, base_name,
                                total_selected_cells=total_selected_cells,
                                series=series)
        print("Ожидание завершения выбора...")
        while not selector.finished:
            plt.pause(0.2)
        print("Ожидание завершено, переходим к следующему изображению.")
        selected_count = len(selector.selected_cells)
        total_selected_cells += selected_count
        print(f"Выбрано клеток на изображении: {selected_count}")
        print(f"Общее количество выбранных клеток в серии: {total_selected_cells}")
        if selected_count > 0:
            selector.calculate_intensities()
            all_results.extend(selector.results)
    if all_results:
        df = pd.DataFrame(all_results)

        # проверяем, что поле существует; если нет — падаем назад на red_intensity_raw
        if sort_param not in df.columns:
            print(f"Поле '{sort_param}' не найдено в данных. Будет использовано 'red_intensity_raw'.")
            sort_col = 'red_intensity_raw'
        else:
            sort_col = sort_param

        df = df.sort_values(by=sort_col, ignore_index=True)
        df['cell_uid'] = df['image'].astype(str) + '__' + df['cell_id'].astype(str)
        
        # вызываем statistical_analysis с sort_col вместо red_intensity_raw
        selected_data, label = statistical_analysis(df, sort_col, folder_path, series)
        
        selected_set = set(selected_data)
        selected_uids = set(df[df[sort_col].isin(selected_set)]['cell_uid'])
        df['selected'] = df['cell_uid'].isin(selected_uids)
        
        df_selected = df[df['selected']]
        
        primary_dir = os.path.join(folder_path, 'PrimaryData')
        os.makedirs(primary_dir, exist_ok=True)
        
        output_csv = os.path.join(primary_dir, f'cell_intensity_results_series_{series}.csv')
        try:
            df.to_csv(output_csv, index=False)
            print(f"Результаты по серии {series} сохранены в файл: {output_csv}")
        except PermissionError:
            print(f"Ошибка: файл {output_csv} занят или защищён от записи!")
        
        calculate_and_save_statistics(df_selected, folder_path, series)
        
        ratio_mean = df_selected['ratio'].mean()
        ratio_sem = df_selected['ratio'].sem()
        print(f"Статистика для серии {series}: ratio = {ratio_mean:.4f} ± {ratio_sem:.4f} (mean ± SEM)")
    else:
        print("Нет выбранных клеток или результатов для сохранения.")

if __name__ == "__main__":
    main()