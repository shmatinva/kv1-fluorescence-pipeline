"""
Main controller script for the Avtocellseek pipeline.

Coordinates the full analysis workflow:
1) runs Cellpose-based segmentation of red-channel images (Object_maker.py),
2) launches interactive cell selection and intensity measurement (cellpose_analysis.py),
3) aggregates per-series CSV files into combined summary tables,
4) performs simple statistical tests and optional normalization of ratio values.

Execute this script to process a folder of TIFF images and obtain per-cell and per-series
intensity ratios (green/red) for downstream analysis of fluorescent binding.
"""
import os
import re
import subprocess
import random
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io, measure
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
import h5py
import math

PROC_DIR = r"C:\Data\Proc"
VENV_DIR = os.path.join(PROC_DIR, "venv")
VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")

OBJECT_MAKER = os.path.join(PROC_DIR, "Object_maker.py")
CELLPOSE_ANALYSIS = os.path.join(PROC_DIR, "cellpose_analysis.py")

def run_script(script_path, args=None):
    cmd = [VENV_PYTHON, script_path]
    if args:
        cmd += args
    print(f"\nЗапуск: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка при запуске {script_path}: {e}")
        return False

def extract_series_number(filename):
    match = re.search(r'_(\d+)_.*_z000_ch01', filename)
    return match.group(1) if match else None

def get_series_order(series_names):
    while True:
        order_input = input("Введите серии для обработки и порядок через запятую (например: 0,1,3): ")
        input_series = [x.strip() for x in order_input.split(',')]
        if all(s in series_names for s in input_series) and input_series:
            return input_series
        print("Ошибка ввода или серии не найдены. Попробуйте снова.")

def create_full_series_order(user_order, all_series_in_data):
    user_order_str = list(map(str, user_order))
    all_series_str = list(map(str, all_series_in_data))
    
    missing_series = sorted(set(all_series_str) - set(user_order_str), key=lambda x: int(x))
    
    full_order = user_order_str + missing_series
    return full_order

def load_and_combine_all_series_data(primary_dir, stats_dir, full_order, sort_param='red_intensity_raw'):
    import glob

    primary_pattern = os.path.join(primary_dir, 'cell_intensity_results_series_*.csv')
    primary_files = glob.glob(primary_pattern)
    if not primary_files:
        print("В папке PrimaryData нет файлов с результатами.")
        all_primary_df = None
    else:
        primary_dfs = []
        for f in primary_files:
            try:
                df = pd.read_csv(f)
                primary_dfs.append(df)
            except Exception as e:
                print(f"Ошибка чтения файла {f}: {e}")
        if primary_dfs:
            all_primary_df = pd.concat(primary_dfs, ignore_index=True)
            all_primary_df['series'] = all_primary_df['series'].astype(str)
            all_primary_df['series'] = pd.Categorical(all_primary_df['series'], categories=full_order, ordered=True)
            all_primary_df = all_primary_df.sort_values(by=['series', sort_param], na_position='last').reset_index(drop=True)
        else:
            all_primary_df = None

    stats_pattern = os.path.join(stats_dir, 'stats_series_*.csv')
    stats_files = glob.glob(stats_pattern)
    if not stats_files:
        print("В папке Statistics нет файлов статистики.")
        all_stats_df = None
    else:
        stats_dfs = []
        for f in stats_files:
            try:
                df = pd.read_csv(f)
                stats_dfs.append(df)
            except Exception as e:
                print(f"Ошибка чтения файла {f}: {e}")
        if stats_dfs:
            all_stats_df = pd.concat(stats_dfs, ignore_index=True)
            all_stats_df['series'] = all_stats_df['series'].astype(str)
            all_stats_df['series'] = pd.Categorical(all_stats_df['series'], categories=full_order, ordered=True)
        else:
            all_stats_df = None

    return all_primary_df, all_stats_df

def compare_green_signal_ttest(all_primary_df, all_series, order, use_norm=False):
    print("Доступные серии для сравнения:", ', '.join(all_series))
    user_series = input("Введите номер интересующей серии для сравнения (одну из перечисленных выше): ").strip()
    if user_series not in all_series:
        print("Серия не найдена в данных.")
        return

    signal_col = 'green_intensity' if not use_norm else 'ratio_norm'
    base = all_primary_df[(all_primary_df['series'].astype(str) == user_series) & (all_primary_df['selected'] == True)][signal_col]

    comparison_results = []
    for s in all_series:
        if s == user_series:
            continue
        other = all_primary_df[(all_primary_df['series'].astype(str) == s) & (all_primary_df['selected'] == True)][signal_col]
        if len(base) > 1 and len(other) > 1:
            tval, pval = ttest_ind(base, other, equal_var=False)
            comparison_results.append({'base_series': user_series, 'compare_series': s, 't_value': tval, 'p_value': pval,
                                      'mean_base': base.mean(), 'mean_compare': other.mean()})
        else:
            comparison_results.append({'base_series': user_series, 'compare_series': s, 't_value': np.nan, 'p_value': np.nan,
                                      'mean_base': base.mean() if len(base) > 0 else np.nan,
                                      'mean_compare': other.mean() if len(other) > 0 else np.nan})
    comp_df = pd.DataFrame(comparison_results)
    print("\nРезультаты теста Стьюдента (green - background):\n")
    print(comp_df)

def ttest_base_series(all_primary_df, base_series, use_norm=False):
    signal_col = 'ratio_norm' if use_norm and 'ratio_norm' in all_primary_df.columns else 'ratio'
    base_vals = pd.to_numeric(all_primary_df[(all_primary_df['series'].astype(str) == base_series) & (all_primary_df['selected'] == True)][signal_col], errors='coerce').dropna()
    results = []
    all_series = sorted(all_primary_df['series'].astype(str).unique())
    for s in all_series:
        if s == base_series:
            continue
        s_vals = pd.to_numeric(all_primary_df[(all_primary_df['series'].astype(str) == s) & (all_primary_df['selected'] == True)][signal_col], errors='coerce').dropna()
        if len(base_vals) > 1 and len(s_vals) > 1:
            tval, pval = ttest_ind(base_vals, s_vals, equal_var=False)
            results.append({'base': base_series, 'compare': s, 't_value': tval, 'p_value': pval,
                            'mean_base': base_vals.mean(), 'mean_compare': s_vals.mean()})
    comp_df = pd.DataFrame(results)
    print("\nРезультаты теста Стьюдента (по параметру {}):".format(signal_col))
    print(comp_df)
    return comp_df

def normalize_stats_after_subtract_base(all_stats_df, base_series, norm_series):
    """
    Нормировка статистики (mean, std, sem) по ratio для каждой серии:
    1) вычитание среднего базовой серии;
    2) нормировка на выбранную серию norm_series;
    3) расчёт комбинированной ошибки для ratio_norm_mean на основе SEM.
    Ожидает в all_stats_df колонки: series, ratio_mean, ratio_sem.
    """
    df = all_stats_df.copy()
    df['series'] = df['series'].astype(str)

    for col in ['ratio_mean', 'ratio_sem']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if base_series is not None:
        base_row = df[df['series'] == str(base_series)]
        if base_row.empty:
            print(f"Базовая серия {base_series} не найдена в статистике. Выполняется без вычитания базы.")
            base_mean = 0
        else:
            base_row = base_row.iloc[0]
            base_mean = base_row['ratio_mean']
    else:
        base_mean = 0

    norm_row = df[df['series'] == str(norm_series)]
    if norm_row.empty:
        print(f"Серия для нормировки {norm_series} не найдена в статистике.")
        return df
    norm_row = norm_row.iloc[0]

    norm_mean = norm_row['ratio_mean']
    norm_sem = norm_row['ratio_sem']

    df['ratio_subtracted_mean'] = df['ratio_mean'] - base_mean
    norm_mean_sub = norm_mean - base_mean

    ratio_norm_means = []
    ratio_norm_errors = []

    def indirect_error_propagation(x, x_err, y, y_err):
        return np.sqrt((x_err / y) ** 2 + (x * y_err / (y ** 2)) ** 2)

    for _, row in df.iterrows():
        x = row['ratio_mean'] - base_mean
        y = norm_mean_sub
        if pd.isna(x) or pd.isna(y) or y == 0:
            ratio_norm_means.append(np.nan)
            ratio_norm_errors.append(np.nan)
            continue

        x_err = row['ratio_sem']
        y_err = norm_sem
        if pd.isna(x_err) or pd.isna(y_err):
            ratio_norm_means.append(np.nan)
            ratio_norm_errors.append(np.nan)
            continue

        norm_ratio = x / y
        norm_error = indirect_error_propagation(x, x_err, y, y_err)

        ratio_norm_means.append(norm_ratio)
        ratio_norm_errors.append(norm_error)

    df['ratio_norm_mean'] = ratio_norm_means
    df['ratio_norm_sem_combined'] = ratio_norm_errors

    print(f"Нормировка статистики по базе {base_series} и нормировочной серии {norm_series} выполнена с использованием SEM.")
    return df

def get_random_files_per_series(folder_path, suffix="_ch01.tif"):
    """
    Возвращает словарь {series_number: случайный файл из серии}
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(suffix)]
    series_dict = {}
    # Группируем файлы по серии
    series_files = {}
    for f in files:
        series = extract_series_number(f)
        if series is None:
            continue
        series_files.setdefault(series, []).append(f)
    # Для каждой серии выбираем случайный файл
    for series, flist in series_files.items():
        series_dict[series] = random.choice(flist)
    return series_dict

def get_short_name(base_name):
    """
    Извлекает короткое имя из base_name.
    base_name пример: '20022025_Kv1-2_Ce4_Hg_G_1_6_z000'
    Нужно получить: 'Kv1-2_Ce4_Hg_G_1_6'
    """
    parts = base_name.split('_')
    if 'z000' in parts:
        z_idx = parts.index('z000')
    else:
        z_idx = len(parts)
    # Пропускаем дату (первый элемент)
    short_name = '_'.join(parts[1:z_idx])
    return short_name
    
def show_random_objects_overlay_one_window(image_folder, object_mask_folder, suffix="_ch01_mask.npy", class_color_dict=None):
    """
    Визуализация случайных изображений с объектными масками — по одному из каждой серии.
    """

    if class_color_dict is None:
        class_color_dict = {1: (1, 0, 0)}  # красный цвет для объектов по умолчанию
    
    series_files = get_random_files_per_series(image_folder, suffix="_ch01.tif")
    sorted_series_files = sorted(series_files.items(), key=lambda x: int(x[0]))

    images = []
    titles = []
    masks = []  # для хранения загруженных масок

    # Первый цикл: загрузка изображений и масок, подготовка их для визуализации
    for i, (series, filename) in enumerate(sorted_series_files):
        base_name = filename[:-len("_ch01.tif")]
        img_path = os.path.join(image_folder, filename)
        mask_path_npy = os.path.join(object_mask_folder, base_name + suffix)
        mask_path_h5 = os.path.join(object_mask_folder, base_name + "_mask.h5")

        # Загружаем маску один раз
        if os.path.exists(mask_path_npy):
            mask = np.load(mask_path_npy)
        elif os.path.exists(mask_path_h5):
            with h5py.File(mask_path_h5, 'r') as f:
                key = list(f.keys())[0]
                mask = np.array(f[key])
        else:
            print(f"Маска не найдена для {filename}, пропускаем.")
            continue

        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.ndim != 2:
            print(f"[LOG] Маска не 2D для {filename} (shape={mask.shape}), пропускаем.")
            continue

        img = io.imread(img_path)
        if mask.shape != img.shape[:2]:
            from skimage.transform import resize
            mask = resize(mask, img.shape[:2], order=0, preserve_range=True).astype(mask.dtype)
            print(f"[DEBUG] Маска приведена к размеру изображения: {mask.shape}")

        if not np.issubdtype(mask.dtype, np.integer):
            mask = mask.astype(np.int32)
            print(f"[DEBUG] Маска приведена к целочисленному типу: {mask.dtype}")

        if img.ndim == 2:
            img = gray2rgb(img)

        # Приведение изображения к float от 0 до 1
        img = img.astype(float)
        if img.max() > 1:
            img /= img.max()

        images.append(img)
        masks.append(mask)  # сохраняем маску
        short_name = get_short_name(base_name)
        titles.append(f"Серия {series}: {short_name}")

    n = len(images)
    if n == 0:
        print("Нет изображений для отображения.")
        return

    cols = min(4, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    figsize = (cols * 4, rows * 4)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()  # для единообразной индексации

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(images[i])
            ax.set_aspect('equal')

            unique_labels = np.unique(masks[i])
            unique_labels = unique_labels[unique_labels != 0]

            for label in unique_labels:
                obj_mask = (masks[i] == label)
                contours = measure.find_contours(obj_mask, level=0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

            ax.set_title(titles[i], fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def categorize_and_sort_df(df, order, sort_param=None, save_path=None):
    """
    Присваивает категориальный тип для столбца 'series' с указанным порядком,
    сортирует DataFrame по series и параметру сортировки (если указан),
    опционально сохраняет отсортированный DataFrame в CSV.
    
    :param df: pandas DataFrame с колонкой 'series'
    :param order: список желаемого порядка серий (категорий)
    :param sort_param: имя столбца, по которому дополнительно сортировать внутри серии (по возрастанию)
    :param save_path: путь для сохранения CSV. Если None, не сохраняет
    :return: отсортированный DataFrame
    """
    order_str = [str(s) for s in order]
    df['series'] = df['series'].astype(str)
    df['series'] = pd.Categorical(df['series'], categories=order_str, ordered=True)
    
    sort_keys = ['series']
    if sort_param and sort_param in df.columns:
        sort_keys.append(sort_param)
        
    df_sorted = df.sort_values(by=sort_keys, na_position='last').reset_index(drop=True)
    
    if save_path:
        df_sorted.to_csv(save_path, index=False)
        print(f"Файл отсортированных данных сохранен: {save_path}")
    
    return df_sorted

def main():
    os.chdir(PROC_DIR)
    print(f"Перешли в рабочую папку: {PROC_DIR}")

    folder_path = input("Введите путь к папке с изображениями: ").strip('"').strip("'")
    if not os.path.isdir(folder_path):
        print("Папка не найдена!")
        return

    stats_dir = os.path.join(folder_path, "Statistics")
    primary_dir = os.path.join(folder_path, "PrimaryData")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(primary_dir, exist_ok=True)

    data_ready = input("Ваши данные полностью обработаны? (да/нет): ").strip().lower()

    processed_series = set()
    series_to_process = []

    # Получаем список всех серий из файлов _ch01.tif
    files = [f for f in os.listdir(folder_path) if f.endswith('_ch01.tif')]
    all_series_names = sorted(set(filter(None, (extract_series_number(f) for f in files))))

    if data_ready == 'да':
        # Если данные готовы, подтягиваем обработанные серии из Statistics
        print("\nПоиск уже обработанных серий в папке Statistics...")
        existing_stats_files = [f for f in os.listdir(stats_dir) if f.startswith('stats_series_') and f.endswith('.csv')]
        existing_series = sorted(set(filter(None, (re.search(r'stats_series_(\d+).csv', f).group(1) if re.search(r'stats_series_(\d+).csv', f) else None for f in existing_stats_files))))

        print("Найденные обработанные серии:", ', '.join(existing_series) if existing_series else "Отсутствуют")

        series_to_process = existing_series

    else:
        # Пользователь сам выбирает серии для обработки — плюс спрашиваем порядок
        print(f"Доступные серии: {all_series_names}")
        order = get_series_order(all_series_names)
        series_to_process = order

    # Если данные полностью обработаны, пропускаем этапы поиска объектов и запуска Object_maker / cellpose
    if data_ready != 'да':
        user_order = order
        # ... Поиск объектов, запуск Object_maker, визуализация ...
        print("\n=== ШАГ 1: Поиск объектов (Object_maker) ===")
        object_mask_folder = os.path.join(folder_path, "Object_masks")

        if os.path.isdir(object_mask_folder) and os.listdir(object_mask_folder):
            print(f"Папка с масками объектов уже существует и не пуста: {object_mask_folder}")
            print("Пропускаем этап поиска объектов.")
        else:
            if not run_script(OBJECT_MAKER, [folder_path]):
                print("Ошибка при запуске Object_maker.")
                return

        print("\nПроверка поиска объектов...")
        show_random_objects_overlay_one_window(folder_path, object_mask_folder, suffix="_ch01_mask.npy")
        cont = input("Готовы начать пользовательскую обработку? (да/нет): ").strip().lower()
        if cont != "да":
            print("Обработка завершена после поиска объектов.")
            return

        print("\n=== ШАГ 2: Пользовательская обработка (cellpose_analysis) ===")
        print("\nВыберите параметр для сортировки клеток внутри серии.")
        print("Доступные варианты: red_intensity_raw, ratio")
        sort_param = input("Введите имя параметра для сортировки: ").strip()
        if sort_param not in ['red_intensity_raw', 'ratio']:
            print("Некорректный параметр, будет использован red_intensity_raw.")
            sort_param = 'red_intensity_raw'

        for series in series_to_process:
            print(f"\nОбрабатываем серию {series}...")
            result = run_script(CELLPOSE_ANALYSIS, [folder_path, series, sort_param])
            if not result:
                print(f"Ошибка при обработке серии {series}")
                continue

            primary_file = os.path.join(primary_dir, f'cell_intensity_results_series_{series}.csv')
            if os.path.exists(primary_file):
                df = pd.read_csv(primary_file)
                processed_series.add(series)
            else:
                print(f"Результаты для серии {series} не найдены.")
                continue

    else:
        # Для готовых данных подтягиваем все результаты, найденные в PrimaryData и Statistics
        user_order = series_to_process
        primary_files = [f for f in os.listdir(primary_dir) if f.startswith('cell_intensity_results_series_') and f.endswith('.csv')]
        primary_series = sorted(set(filter(None, (re.search(r'cell_intensity_results_series_(\d+).csv', f).group(1) if re.search(r'cell_intensity_results_series_(\d+).csv', f) else None for f in primary_files))))

        print("Найденные обработанные серии:", ', '.join(primary_series) if primary_series else "Отсутствуют")

        for s in primary_series:
            pf = os.path.join(primary_dir, f'cell_intensity_results_series_{s}.csv')
            df = pd.read_csv(pf)
            processed_series.add(s)

        sort_param = input("Введите параметр для сортировки итогового объединения (red_intensity_raw/ratio): ").strip()
        if sort_param not in ['red_intensity_raw', 'ratio']:
            print("Некорректный параметр, будет использован red_intensity_raw.")
            sort_param = 'red_intensity_raw'

    # Далее объединяем и сохраняем результаты по всем обработанным сериям для итогового файла
    all_series_in_data = set(all_series_names)  # все серии из входных файлов
    full_order = create_full_series_order(user_order, all_series_in_data)
    all_primary_df, all_stats_df = load_and_combine_all_series_data(primary_dir, stats_dir, full_order, sort_param=sort_param)
    
    if all_primary_df is not None:
        combined_csv = os.path.join(primary_dir, 'all_series_combined_sorted.csv')
        filtered_df = all_primary_df[all_primary_df['selected'] == True]
        filtered_df.to_csv(combined_csv, index=False)
        print(f"\nОбщие результаты сохранены в {combined_csv}")
    
        all_primary_data_path = os.path.join(primary_dir, 'all_primary_data.csv')
        all_primary_df.to_csv(all_primary_data_path, index=False)
    
    if all_stats_df is not None:
        all_stats_df = categorize_and_sort_df(all_stats_df, full_order)
        stats_raw_path = os.path.join(stats_dir, 'all_series_statistics_raw.csv')
        all_stats_df.to_csv(stats_raw_path, index=False)
        print(f"Общая статистика (до нормировки) сохранена в {stats_raw_path}")

    if all_primary_df is not None:
        print("\n=== Шаг 3: Постобработка ===")
    
        all_series = list(all_primary_df['series'].cat.categories) if hasattr(all_primary_df['series'], 'cat') else sorted(all_primary_df['series'].astype(str).unique())
    
        # Запрос базовой серии для t-теста с выводом
        print("Доступные серии для базового сравнения:", ', '.join(all_series))
        base_series = input("Введите номер базовой серии для сравнения (введите 'нет', чтобы пропустить): ").strip()
    
        if base_series == 'нет':
            print("Пропускаем тест Стьюдента и вычитание базовой серии.")
        elif base_series not in all_series:
            print("Базовая серия указана неверно. Обработка завершена.")
            return
        else:
            # Запуск t-теста
            ttest_base_series(all_primary_df, base_series, use_norm=False)
    
        # Запрос необходимости нормировки (упрощённый)
        normalize_answer = input("Выполнить нормировку? (да/нет): ").strip().lower()
        if normalize_answer == 'да':
            print("Доступные серии для нормировки:", ', '.join(all_series))
            norm_series = input("Введите номер серии для нормировки: ").strip()
            if norm_series not in all_series:
                print("Серия для нормировки некорректна. Нормировка пропущена.")
            else:
                if all_stats_df is not None:
                    base_series_for_norm = base_series if base_series != 'нет' else None
                    norm_stats_df = normalize_stats_after_subtract_base(all_stats_df, base_series_for_norm, norm_series)
                    norm_stats_path = os.path.join(stats_dir, 'all_series_statistics_norm.csv')
                    norm_stats_df.to_csv(norm_stats_path, index=False)
                    print(f"Нормированная статистика сохранена в {norm_stats_path}")
                    print("\nНормированные значения по сериям (ratio_norm_mean, ratio_norm_sem_combined):")
                    print(norm_stats_df[['series', 'ratio_norm_mean', 'ratio_norm_sem_combined']].to_string(index=False))
        else:
            print("Пропускаем нормировку.")
    
    else:
        print("Нет обработанных серий, финализировать нечего.")
    
    print("\n=== Конвейер завершён ===")

if __name__ == "__main__":
    main()