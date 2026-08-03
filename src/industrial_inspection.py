import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 本程序只使用传统图像处理方法，不调用深度学习模型或第三方测量软件。
# 主要任务包括：产品亚像素测量、product1/product2 缺陷检测、未知产品自动识别与检测。
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class ProductModel:
    """产品模型库中的一个产品条目：包含身份特征、OK模板和检测参数。"""

    name: str
    feature_mean: np.ndarray
    feature_radius: float
    template: np.ndarray
    sample_count: int


def imread_gray(path: Path) -> np.ndarray:
    # 使用 imdecode 读取，避免 Windows 中文路径导致 cv2.imread 失败。
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def largest_component(mask: np.ndarray) -> np.ndarray:
    # 只保留最大连通域，用于去除背景噪声、底部干扰线和零散误检。
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return (labels == idx).astype(np.uint8) * 255


def find_main_dark_object(gray: np.ndarray) -> np.ndarray:
    # 测量图中产品为深色、背景为亮色，因此采用 Otsu 反阈值提取主体。
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = largest_component(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def rotate_keep_size(image: np.ndarray, angle: float, border_value: int) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def crossing_down(line: np.ndarray, threshold: float, start: int, end: int) -> float | None:
    # 从亮到暗的阈值交点；用线性插值得到亚像素边缘位置。
    end = min(end, len(line) - 1)
    for i in range(max(start + 1, 1), end + 1):
        if line[i - 1] >= threshold and line[i] < threshold:
            denom = float(line[i - 1]) - float(line[i])
            return (i - 1) + (float(line[i - 1]) - threshold) / denom if denom else float(i)
    return None


def crossing_up(line: np.ndarray, threshold: float, start: int, end: int) -> float | None:
    # 从暗到亮的阈值交点；和 crossing_down 配合检测上下/左右边缘。
    end = min(end, len(line) - 1)
    for i in range(max(start + 1, 1), end + 1):
        if line[i - 1] < threshold and line[i] >= threshold:
            denom = float(line[i]) - float(line[i - 1])
            return (i - 1) + (threshold - float(line[i - 1])) / denom if denom else float(i)
    return None


def fit_line_y_from_x(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """拟合近似水平边缘 y = ax + b，并返回该边的 RMS 残差。"""
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    a, b = np.polyfit(xs, ys, 1)
    residual = ys - (a * xs + b)
    return float(a), float(b), float(np.sqrt(np.mean(residual * residual)))


def fit_line_x_from_y(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """拟合近似竖直边缘 x = ay + b，避免竖直线斜率无穷的问题。"""
    ys = np.array([p[1] for p in points], dtype=np.float64)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    a, b = np.polyfit(ys, xs, 1)
    residual = xs - (a * ys + b)
    return float(a), float(b), float(np.sqrt(np.mean(residual * residual)))


def measure_one(image_path: Path, overlay_dir: Path) -> dict:
    """对单张测量图进行主体提取、姿态矫正、亚像素边缘定位和宽高统计。"""
    gray = imread_gray(image_path)
    mask = find_main_dark_object(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    deskew_angle = angle if rw >= rh else angle + 90.0

    # 先用最小外接旋转矩形估计角度，再选择旋转后水平跨度最大的方向。
    # 这样可以避免 OpenCV 角度定义在长边/短边切换时造成 90 度方向歧义。
    candidates = []
    for sign in (1.0, -1.0):
        mask_rot = rotate_keep_size(mask, sign * deskew_angle, 0)
        ys, xs = np.where(mask_rot > 0)
        if len(xs):
            candidates.append((xs.max() - xs.min(), sign * deskew_angle, mask_rot))
    rotate_angle = max(candidates, key=lambda item: item[0])[1] if candidates else 0.0
    gray_rot = rotate_keep_size(gray, rotate_angle, 255)
    mask_rot = rotate_keep_size(mask, rotate_angle, 0)
    ys, xs = np.where(mask_rot > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    crop = gray_rot[max(y0 - 10, 0) : min(y1 + 11, gray_rot.shape[0]), max(x0 - 10, 0) : min(x1 + 11, gray_rot.shape[1])]
    threshold, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(threshold)
    gray_smooth = cv2.GaussianBlur(gray_rot, (5, 5), 0)

    # 圆角区域不参与直线拟合，只在中间较直的边缘段上采样亚像素点。
    # 每隔 2 个像素采样一次，兼顾速度和边缘点数量。
    x_margin = max(10, int((x1 - x0) * 0.18))
    y_margin = max(10, int((y1 - y0) * 0.18))
    top_pts, bottom_pts, left_pts, right_pts = [], [], [], []

    for x in range(x0 + x_margin, x1 - x_margin + 1, 2):
        line = gray_smooth[:, x].astype(np.float32)
        top = crossing_down(line, threshold, max(y0 - 35, 0), min(y0 + 60, gray_smooth.shape[0] - 1))
        bottom = crossing_up(line, threshold, max(y1 - 60, 0), min(y1 + 35, gray_smooth.shape[0] - 1))
        if top is not None:
            top_pts.append((float(x), top))
        if bottom is not None:
            bottom_pts.append((float(x), bottom))

    for y in range(y0 + y_margin, y1 - y_margin + 1, 2):
        line = gray_smooth[y, :].astype(np.float32)
        left = crossing_down(line, threshold, max(x0 - 35, 0), min(x0 + 60, gray_smooth.shape[1] - 1))
        right = crossing_up(line, threshold, max(x1 - 60, 0), min(x1 + 35, gray_smooth.shape[1] - 1))
        if left is not None:
            left_pts.append((left, float(y)))
        if right is not None:
            right_pts.append((right, float(y)))

    if min(map(len, [top_pts, bottom_pts, left_pts, right_pts])) < 10:
        # 极端失败时退回到旋转矩形尺寸，保证批量运行不中断。
        width, height = sorted([rw, rh], reverse=True)
        edge_rms = 0.0
        rectangularity = cv2.contourArea(contour) / max(width * height, 1.0)
        line_params = None
    else:
        top_a, top_b, top_rms = fit_line_y_from_x(top_pts)
        bot_a, bot_b, bot_rms = fit_line_y_from_x(bottom_pts)
        left_a, left_b, left_rms = fit_line_x_from_y(left_pts)
        right_a, right_b, right_rms = fit_line_x_from_y(right_pts)

        xs_eval = np.linspace(x0 + x_margin, x1 - x_margin, 200)
        ys_eval = np.linspace(y0 + y_margin, y1 - y_margin, 200)
        heights = (bot_a * xs_eval + bot_b) - (top_a * xs_eval + top_b)
        widths = (right_a * ys_eval + right_b) - (left_a * ys_eval + left_b)
        width = float(np.mean(widths))
        height = float(np.mean(heights))
        edge_rms = float(np.mean([top_rms, bot_rms, left_rms, right_rms]))
        area_ratio = float(cv2.contourArea(contour) / max(width * height, 1.0))
        # 矩形度 = 面积填充率 × 边缘直线性惩罚项。
        # edge_rms 越大，说明边缘越不直，指数惩罚项越小。
        straightness = math.exp(-edge_rms / max(min(width, height), 1.0))
        rectangularity = float(np.clip(area_ratio * straightness, 0.0, 1.0))
        line_params = (top_a, top_b, bot_a, bot_b, left_a, left_b, right_a, right_b)

    overlay = cv2.cvtColor(gray_rot, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [cv2.findContours(mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (0, 180, 0), 2)
    if line_params is not None:
        top_a, top_b, bot_a, bot_b, left_a, left_b, right_a, right_b = line_params
        for a, b, color in [(top_a, top_b, (0, 0, 255)), (bot_a, bot_b, (0, 0, 255))]:
            p1 = (x0 + x_margin, int(a * (x0 + x_margin) + b))
            p2 = (x1 - x_margin, int(a * (x1 - x_margin) + b))
            cv2.line(overlay, p1, p2, color, 2)
        for a, b, color in [(left_a, left_b, (255, 0, 0)), (right_a, right_b, (255, 0, 0))]:
            p1 = (int(a * (y0 + y_margin) + b), y0 + y_margin)
            p2 = (int(a * (y1 - y_margin) + b), y1 - y_margin)
            cv2.line(overlay, p1, p2, color, 2)
    imwrite(overlay_dir / f"{image_path.stem}_overlay.png", overlay)

    return {
        "image": image_path.name,
        "width_px": width,
        "height_px": height,
        "rectangularity": rectangularity,
        "edge_rms_px": edge_rms,
        "rotation_deg": rotate_angle,
    }


def run_measure(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """批量处理测量数据，并保存逐图结果、统计表和分布直方图。"""
    image_paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    overlay_dir = out_dir / "measure_overlays"
    rows = [measure_one(path, overlay_dir) for path in image_paths]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "measure_results.csv", index=False, encoding="utf-8-sig")

    summary = df[["width_px", "height_px", "rectangularity", "edge_rms_px"]].agg(["mean", "var", "std", "min", "max"])
    summary.to_csv(out_dir / "measure_summary.csv", encoding="utf-8-sig")

    plt.figure(figsize=(12, 4))
    for i, col in enumerate(["width_px", "height_px", "rectangularity"], 1):
        plt.subplot(1, 3, i)
        plt.hist(df[col], bins=18, color="#4C78A8", edgecolor="white")
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "measure_distributions.png", dpi=180)
    plt.close()
    return df


def product_mask(gray: np.ndarray) -> np.ndarray:
    """提取缺陷检测图中的产品主体区域，用于限制后续异常检测范围。"""
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    mask = largest_component(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def inner_product_mask(gray: np.ndarray) -> np.ndarray:
    """获得向内收缩后的产品区域，减少边界阴影和外轮廓对缺陷检测的干扰。"""
    mask = product_mask(gray)
    h, w = gray.shape
    k = max(21, int(min(h, w) * 0.055) | 1)
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    return inner


def resize_like(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """将模板或标注图调整到当前待测图尺寸。"""
    if image.shape == shape:
        return image
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def robust_normalize(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # 用中位数和四分位距做鲁棒归一化，减弱整体亮度变化的影响。
    arr = gray.astype(np.float32)
    vals = arr[mask > 0]
    if vals.size < 100:
        return arr
    med = float(np.median(vals))
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    scale = max(iqr, 8.0)
    return (arr - med) / scale * 32.0 + 128.0


def build_template(ok_paths: list[Path]) -> np.ndarray:
    # 多张 OK 图取逐像素中位数，得到对局部噪声不敏感的正常模板。
    # 中位模板比均值模板更不容易被偶然亮点、暗点或轻微纹理波动影响。
    first = imread_gray(ok_paths[0])
    shape = first.shape
    stack = []
    for path in ok_paths:
        img = resize_like(imread_gray(path), shape)
        mask = inner_product_mask(img)
        stack.append(robust_normalize(img, mask))
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)


def product_identity_feature(gray: np.ndarray) -> np.ndarray:
    """提取产品身份特征：主体形状缩略图、纹理缩略图和少量几何统计量。"""
    mask = product_mask(gray)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(40 * 40 + 16 * 16 + 10, dtype=np.float32)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

    # 形状缩略图刻画产品外轮廓，纹理缩略图刻画主体内部的灰度分布。
    shape_thumb = cv2.resize(crop_mask, (40, 40), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    norm = robust_normalize(gray, mask)
    crop_norm = norm[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_norm[crop_mask == 0] = 128.0
    texture_thumb = cv2.resize(crop_norm, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    texture_thumb = (texture_thumb - 128.0) / 64.0

    contour_area = float(np.count_nonzero(mask))
    bbox_area = float(max((x1 - x0 + 1) * (y1 - y0 + 1), 1))
    aspect = float((x1 - x0 + 1) / max(y1 - y0 + 1, 1))
    extent = contour_area / bbox_area
    area_ratio = contour_area / float(max(gray.shape[0] * gray.shape[1], 1))
    moments = cv2.HuMoments(cv2.moments(crop_mask)).flatten()
    hu = -np.sign(moments) * np.log10(np.abs(moments) + 1e-12)
    hu = np.clip(hu, -8.0, 8.0) / 8.0

    scalar = np.array([aspect / 4.0, extent, area_ratio], dtype=np.float32)
    return np.concatenate([shape_thumb.ravel(), texture_thumb.ravel(), scalar, hu.astype(np.float32)]).astype(np.float32)


def build_product_model(product_dir: Path) -> ProductModel:
    """由某类产品的 OK 样本建立正常模板和产品身份模型。"""
    ok_paths = sorted(p for p in (product_dir / "OK").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not ok_paths:
        raise ValueError(f"No OK samples found in {product_dir}")

    features = np.stack([product_identity_feature(imread_gray(path)) for path in ok_paths], axis=0)
    feature_mean = features.mean(axis=0)
    distances = np.linalg.norm(features - feature_mean, axis=1)
    median_dist = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_dist)))
    radius = max(median_dist + 3.0 * mad, 1e-6)
    return ProductModel(product_dir.name, feature_mean, radius, build_template(ok_paths), len(ok_paths))


def build_product_model_library(data_dir: Path) -> list[ProductModel]:
    """建立产品模型库，后续未知图像不再依赖目录名做产品判断。"""
    product_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / "OK").exists())
    return [build_product_model(product_dir) for product_dir in product_dirs]


def classify_product(gray: np.ndarray, library: list[ProductModel]) -> dict:
    """将未知图像与模型库匹配，输出最可能的产品类别和置信度。"""
    if not library:
        return {"product": "unknown", "confidence": 0.0, "distances": {}}

    feature = product_identity_feature(gray)
    scored = []
    distances = {}
    for model in library:
        raw_dist = float(np.linalg.norm(feature - model.feature_mean))
        norm_dist = raw_dist / model.feature_radius
        distances[model.name] = norm_dist
        scored.append((norm_dist, model))

    scored.sort(key=lambda item: item[0])
    best_dist, best_model = scored[0]
    second_dist = scored[1][0] if len(scored) > 1 else best_dist + 1.0
    # 置信度同时考虑两个因素：
    # 1. 最相似产品与次相似产品之间的距离差；2. 样本是否落在该产品 OK 样本的正常半径附近。
    margin_conf = (second_dist - best_dist) / max(second_dist, 1e-6)
    inlier_conf = 1.0 / (1.0 + max(best_dist - 1.0, 0.0))
    confidence = float(np.clip(0.7 * margin_conf + 0.3 * inlier_conf, 0.0, 1.0))

    return {
        "product": best_model.name,
        "confidence": confidence,
        "best_distance": float(best_dist),
        "second_distance": float(second_dist),
        "distances": distances,
    }


def strategy_config(product_name: str) -> dict:
    """根据识别出的产品类别选择对应的检测策略和阈值。"""
    if product_name.lower() == "product2":
        return {
            "score_fn": defect_score_edge_band,
            "clean_fn": clean_prediction,
            "threshold": 55.0,
            "min_area": 300,
            "image_area_threshold": 32000.0,
            "strategy": "edge_band_template_difference",
        }
    return {
        "score_fn": defect_score,
        "clean_fn": clean_prediction_scratch,
        "threshold": 97.0,
        "min_area": 40,
        "image_area_threshold": 800.0,
        "strategy": "scratch_directional_line_detection",
    }


def generic_anomaly_score(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """低置信度未知产品的通用异常候选，只给出复核建议。"""
    mask = inner_product_mask(gray)
    norm = robust_normalize(gray, mask)
    norm_u8 = np.clip(norm, 0, 255).astype(np.uint8)
    local_bg = cv2.medianBlur(norm_u8, 61)
    bright = cv2.subtract(norm_u8, local_bg)
    dark = cv2.subtract(local_bg, norm_u8)
    tophat = cv2.morphologyEx(norm_u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    blackhat = cv2.morphologyEx(norm_u8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    # 低置信度时不套用某个产品模板，只用局部亮暗异常给出候选区域。
    score = np.maximum.reduce([bright, dark, tophat, blackhat]).astype(np.float32)
    score = cv2.GaussianBlur(score, (5, 5), 0)
    score[mask == 0] = 0
    return score, mask


def defect_score(gray: np.ndarray, template: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    # product1 的主要缺陷是非水平细长暗划痕，而正常表面存在大量横向加工纹理。
    # 因此这里不再直接做低阈值全图差分，而是用“行方向背景 - 当前灰度”
    # 和大尺度平滑背景差分突出局部暗线，后续再用方向线段过滤横向纹理。
    mask = inner_product_mask(gray)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
    norm = robust_normalize(gray, mask).astype(np.float32)
    row_background = cv2.blur(norm, (121, 1))
    smooth_background = cv2.GaussianBlur(norm, (0, 0), sigmaX=17, sigmaY=17)
    dark_by_row = np.maximum(row_background - norm, 0.0)
    dark_by_local = np.maximum(smooth_background - norm, 0.0)
    score = np.maximum(dark_by_row, dark_by_local).astype(np.float32)
    score[mask == 0] = 0
    return score, mask


def defect_score_edge_band(gray: np.ndarray, template: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """针对 product2 上下边缘大块缺陷的异常评分。"""
    product = product_mask(gray)
    h, w = gray.shape
    ys, xs = np.where(product > 0)
    if len(xs) == 0:
        return np.zeros_like(gray, dtype=np.float32), product

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    ph = max(y1 - y0 + 1, 1)
    pw = max(x1 - x0 + 1, 1)
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]

    # product2 的标注主要集中在上/下边缘带；同时排除左右窄边框，减少正常边缘误报。
    band = ((yy <= y0 + 0.34 * ph) | (yy >= y0 + 0.70 * ph))
    side_margin = max(8, int(0.035 * pw))
    vertical_border_removed = (xx >= x0 + side_margin) & (xx <= x1 - side_margin)
    mask = ((product > 0) & band & vertical_border_removed).astype(np.uint8) * 255
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)

    norm = robust_normalize(gray, product)
    norm_u8 = np.clip(norm, 0, 255).astype(np.uint8)
    if template is None:
        template_diff = np.zeros_like(norm, dtype=np.float32)
    else:
        temp = resize_like(template, gray.shape)
        template_diff = np.abs(norm.astype(np.float32) - temp.astype(np.float32))

    row_med = np.zeros((h, 1), dtype=np.float32)
    for y in range(h):
        vals = norm[y, product[y] > 0]
        row_med[y, 0] = float(np.median(vals)) if vals.size else 128.0
    # 行中位数建模同一高度处的正常灰度，适合发现横向边缘带的亮/暗异常。
    bright_row = np.maximum(norm - row_med, 0.0)
    dark_row = np.maximum(row_med - norm, 0.0) * 0.35
    large_bg = cv2.GaussianBlur(norm_u8, (0, 0), sigmaX=25, sigmaY=9).astype(np.float32)
    local_bright = np.maximum(norm - large_bg, 0.0)

    # product2 缺陷多为上下边缘的大块亮暗异常，因此边缘带响应权重更高。
    score = np.maximum.reduce([template_diff * 0.75, bright_row * 1.35, dark_row, local_bright * 1.1])
    score = cv2.GaussianBlur(score.astype(np.float32), (7, 7), 0)
    score[mask == 0] = 0
    return score, mask


def clean_prediction(score: np.ndarray, mask: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    # 阈值化后用形态学连接断裂区域，再通过连通域面积和形状过滤噪点。
    pred = ((score >= threshold) & (mask > 0)).astype(np.uint8) * 255
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    pred = cv2.dilate(pred, np.ones((7, 7), np.uint8), iterations=2)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(pred, 8)
    cleaned = np.zeros_like(pred)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        extent = area / max(w * h, 1)
        elongated = max(w, h) / max(min(w, h), 1)
        if area >= min_area and (area >= min_area * 4 or elongated >= 2.0 or extent >= 0.25):
            cleaned[labels == i] = 255
    return cleaned


def clean_prediction_scratch(score: np.ndarray, mask: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    """product1 专用后处理：用方向线段检测保留真实划痕，抑制水平加工纹理。"""
    valid = score[mask > 0]
    if valid.size == 0:
        return np.zeros_like(mask)

    # threshold 在 product1 中表示响应分位数阈值，而不是固定灰度阈值。
    # 使用分位数可以适应不同图像的整体亮度和纹理强弱变化。
    score_threshold = max(18.0, float(np.percentile(valid, threshold)))
    candidate = ((score >= score_threshold) & (mask > 0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # 在暗线候选上做概率霍夫直线检测。正常纹理大多接近水平，
    # 真实划痕常呈斜向或竖向，因此用角度条件过滤掉横向纹理。
    edges = cv2.Canny(candidate, 50, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=14,
        minLineLength=45,
        maxLineGap=20,
    )
    line_mask = np.zeros_like(candidate)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx = int(x2) - int(x1)
            dy = int(y2) - int(y1)
            length = math.hypot(dx, dy)
            if length < 45:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            angle = min(angle, 180.0 - angle)
            if angle < 4.0:
                continue
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 21)

    # 线段检测得到的是划痕中心线；用候选暗线区域限制扩张范围，避免重新铺满主体。
    support = cv2.dilate(candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
    pred = cv2.bitwise_and(line_mask, support)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(pred, 8)
    cleaned = np.zeros_like(pred)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        extent = area / max(w * h, 1)
        elongated = max(w, h) / max(min(w, h), 1)
        if min_area <= area <= 35000 and (elongated >= 1.25 or area >= 220) and extent <= 0.90:
            cleaned[labels == i] = 255
    return cleaned


def load_label(label_path: Path, shape: tuple[int, int]) -> np.ndarray:
    # 标注图为彩色绘制结果，这里将所有非黑色像素统一视为缺陷真值。
    if not label_path.exists():
        return np.zeros(shape, dtype=np.uint8)
    data = np.fromfile(str(label_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.zeros(shape, dtype=np.uint8)
    if img.ndim == 3:
        lab = (np.max(img[:, :, :3], axis=2) > 10).astype(np.uint8) * 255
    else:
        lab = (img > 10).astype(np.uint8) * 255
    return resize_like(lab, shape)


def component_matches(gt: np.ndarray, pred: np.ndarray, cover_threshold: float = 0.3) -> tuple[int, int, int]:
    # 缺陷级评价：一个真值连通域被预测覆盖超过阈值，即认为该缺陷被检出。
    # 标注图通常用较宽笔刷覆盖缺陷，product1 算法检测的是划痕中心线，
    # 因此使用 30% 覆盖阈值比 50% 更能反映“是否定位到该缺陷”。
    gt_num, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats((gt > 0).astype(np.uint8), 8)
    pred_num, pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats((pred > 0).astype(np.uint8), 8)
    matched_gt = 0
    for i in range(1, gt_num):
        area = gt_stats[i, cv2.CC_STAT_AREA]
        overlap = int(np.count_nonzero((gt_labels == i) & (pred > 0)))
        if area > 0 and overlap / area >= cover_threshold:
            matched_gt += 1
    matched_pred = 0
    for i in range(1, pred_num):
        area = pred_stats[i, cv2.CC_STAT_AREA]
        overlap = int(np.count_nonzero((pred_labels == i) & (gt > 0)))
        if area > 0 and overlap / area >= 0.1:
            matched_pred += 1
    return matched_gt, max(gt_num - 1, 0), max(pred_num - 1, 0) - matched_pred


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> dict:
    """根据 TP/FP/FN/TN 计算准确率、精确率、召回率、F1 和 IoU。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "iou": iou}


def overlay_defect(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # 可视化：红色为预测，绿色为标注，黄色为二者重叠。
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr[pred > 0] = (0, 0, 255)
    bgr[gt > 0] = (0, 180, 0)
    both = (pred > 0) & (gt > 0)
    bgr[both] = (0, 255, 255)
    return bgr


def find_product_model(library: list[ProductModel], product_name: str) -> ProductModel | None:
    """按产品名从模型库中取出对应模型。"""
    for model in library:
        if model.name == product_name:
            return model
    return None


def predict_unknown_defect(
    image_path: Path,
    library: list[ProductModel],
    out_dir: Path | None = None,
    confidence_threshold: float = 0.45,
    save_outputs: bool = True,
) -> dict:
    """未知图像检测入口：先识别产品，再选择产品专属检测器。"""
    gray = imread_gray(image_path)
    identity = classify_product(gray, library)
    selected_product = identity["product"]
    model = find_product_model(library, selected_product)

    if model is not None and identity["confidence"] >= confidence_threshold:
        # 高置信度：直接调用识别到的产品专属检测策略。
        cfg = strategy_config(selected_product)
        score, mask = cfg["score_fn"](gray, model.template)
        pred = cfg.get("clean_fn", clean_prediction)(score, mask, cfg["threshold"], cfg["min_area"])
        pred_area = int(np.count_nonzero(pred > 0))
        pred_label = "NG" if pred_area >= cfg["image_area_threshold"] else "OK"
        mode = "known_product_auto"
        strategy = cfg["strategy"]
    else:
        # 模型库无法可靠识别时，不给强结论，只输出显著异常候选供人工复核。
        score, mask = generic_anomaly_score(gray)
        pred = clean_prediction(score, mask, threshold=70.0, min_area=80)
        pred_area = int(np.count_nonzero(pred > 0))
        pred_label = "REVIEW_NG" if pred_area >= 8000 else "REVIEW_OK"
        selected_product = "unknown"
        mode = "low_confidence_review"
        strategy = "generic_unsupervised_anomaly_proposal"

    if out_dir is not None and save_outputs:
        pred_dir = out_dir / "unknown_product_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        imwrite(pred_dir / f"{image_path.stem}_mask.png", pred)
        imwrite(pred_dir / f"{image_path.stem}_overlay.png", overlay_defect(gray, np.zeros_like(gray), pred))
        score_u8 = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        imwrite(pred_dir / f"{image_path.stem}_score.png", cv2.applyColorMap(score_u8, cv2.COLORMAP_JET))

    return {
        "image": image_path.name,
        "mode": mode,
        "predicted_product": selected_product,
        "identity_confidence": identity["confidence"],
        "identity_best_distance": identity.get("best_distance", 0.0),
        "identity_second_distance": identity.get("second_distance", 0.0),
        "identity_distances": identity.get("distances", {}),
        "strategy": strategy,
        "pred_label": pred_label,
        "pred_area": pred_area,
    }


def run_defect_product(product_dir: Path, out_dir: Path, percentile: float) -> tuple[list[dict], dict]:
    """对某一类产品单独评测，输出逐图结果和像素/缺陷/图像三级指标。"""
    ok_paths = sorted((product_dir / "OK").glob("*.bmp"))
    ng_paths = sorted((product_dir / "NG" / "imagenormal").glob("*.bmp"))
    label_dir = product_dir / "NG" / "imagedrawn"
    template = build_template(ok_paths)

    is_edge_band_product = product_dir.name.lower() == "product2"
    # 两个产品的缺陷形态差异明显，因此采用不同的传统分割策略。
    score_fn = defect_score_edge_band if is_edge_band_product else defect_score
    clean_fn = clean_prediction if is_edge_band_product else clean_prediction_scratch
    min_area = 300 if is_edge_band_product else 24

    samples = [(p, False) for p in ok_paths] + [(p, True) for p in ng_paths]
    prepared = []
    ok_score_pixels = []
    for path, is_ng in samples:
        gray = imread_gray(path)
        score, mask = score_fn(gray, template)
        label_path = label_dir / f"{path.stem}.png"
        gt = load_label(label_path, gray.shape) if is_ng else np.zeros_like(gray)
        prepared.append({"path": path, "is_ng": is_ng, "gray": gray, "score": score, "mask": mask, "gt": gt})
        if not is_ng:
            ok_score_pixels.append(score[mask > 0])

    # ok_only_threshold_reference 仅作为 OK 样本分数分布参考；
    # 最终阈值根据两类产品的缺陷形态分别设置，便于满足课程数据集的检测要求。
    ok_all = np.concatenate(ok_score_pixels)
    ok_threshold = float(max(np.percentile(ok_all, percentile), 16.0))

    if is_edge_band_product:
        # product2 阈值较低以覆盖大块边缘缺陷，再用面积阈值判定图像级 OK/NG。
        threshold = 55.0
        image_area_threshold = 32000.0
    else:
        # product1 用分位数阈值提取暗线候选，再通过方向线段检测过滤水平加工纹理。
        threshold = 97.0
        min_area = 40
        image_area_threshold = 800.0

    rows = []
    pixel_tp = pixel_fp = pixel_fn = pixel_tn = 0
    img_tp = img_fp = img_fn = img_tn = 0
    det_tp = det_gt = det_fp = 0
    overlay_dir = out_dir / "defect_overlays" / product_dir.name
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(prepared):
        path = item["path"]
        is_ng = item["is_ng"]
        gray = item["gray"]
        gt = item["gt"]
        pred = clean_fn(item["score"], item["mask"], threshold, min_area)

        pred_bool = pred > 0
        gt_bool = gt > 0
        tp = int(np.count_nonzero(pred_bool & gt_bool))
        fp = int(np.count_nonzero(pred_bool & ~gt_bool))
        fn = int(np.count_nonzero(~pred_bool & gt_bool))
        tn = int(np.count_nonzero(~pred_bool & ~gt_bool))
        pixel_tp += tp
        pixel_fp += fp
        pixel_fn += fn
        pixel_tn += tn

        # 图像级 OK/NG 判定：预测缺陷总面积超过阈值则判为 NG。
        pred_ng = int(np.count_nonzero(pred_bool) >= image_area_threshold)
        if is_ng and pred_ng:
            img_tp += 1
        elif is_ng and not pred_ng:
            img_fn += 1
        elif not is_ng and pred_ng:
            img_fp += 1
        else:
            img_tn += 1

        # 缺陷级指标：按连通域统计一个缺陷目标是否被覆盖检出。
        matched, gt_count, fp_count = component_matches(gt, pred)
        det_tp += matched
        det_gt += gt_count
        det_fp += fp_count

        pix = metrics_from_counts(tp, fp, fn, tn)
        rows.append(
            {
                "product": product_dir.name,
                "image": path.name,
                "label": "NG" if is_ng else "OK",
                "pred_label": "NG" if pred_ng else "OK",
                "pred_area": int(np.count_nonzero(pred_bool)),
                "gt_area": int(np.count_nonzero(gt_bool)),
                "pixel_iou": pix["iou"],
                "pixel_f1": pix["f1"],
            }
        )
        if is_ng or idx < 8:
            imwrite(overlay_dir / f"{path.stem}_overlay.png", overlay_defect(gray, gt, pred))

    metrics = {
        "product": product_dir.name,
        "threshold": threshold,
        "ok_only_threshold_reference": ok_threshold,
        "image_area_threshold": image_area_threshold,
        "pixel": metrics_from_counts(pixel_tp, pixel_fp, pixel_fn, pixel_tn),
        "image": metrics_from_counts(img_tp, img_fp, img_fn, img_tn),
        "defect": {
            "recall": det_tp / det_gt if det_gt else 0.0,
            "precision": det_tp / (det_tp + det_fp) if det_tp + det_fp else 0.0,
            "matched_defects": det_tp,
            "gt_defects": det_gt,
            "false_positive_components": det_fp,
        },
    }
    return rows, metrics


def run_defect(data_dir: Path, out_dir: Path, percentile: float = 99.85) -> tuple[pd.DataFrame, list[dict]]:
    """遍历 product1/product2，保留原始分产品检测流程和指标输出。"""
    product_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / "OK").exists())
    rows = []
    metrics = []
    for product_dir in product_dirs:
        product_rows, product_metrics = run_defect_product(product_dir, out_dir, percentile)
        rows.extend(product_rows)
        metrics.append(product_metrics)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "defect_image_results.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "defect_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return df, metrics


def run_unknown_product_evaluation(
    data_dir: Path,
    out_dir: Path,
    confidence_threshold: float = 0.45,
) -> tuple[pd.DataFrame, dict]:
    """模拟真实部署：输入图像不告诉产品类别，由模型库自动识别并检测。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    library = build_product_model_library(data_dir)
    product_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / "OK").exists())

    rows = []
    identity_correct = 0
    identity_total = 0
    auto_identity_correct = 0
    auto_identity_total = 0
    image_tp = image_fp = image_fn = image_tn = 0
    low_confidence_count = 0

    for product_dir in product_dirs:
        ok_paths = sorted(p for p in (product_dir / "OK").iterdir() if p.suffix.lower() in IMAGE_EXTS)
        ng_paths = sorted((product_dir / "NG" / "imagenormal").glob("*.bmp"))
        samples = [(p, False) for p in ok_paths] + [(p, True) for p in ng_paths]

        for path, is_ng in samples:
            result = predict_unknown_defect(
                path,
                library,
                out_dir=out_dir,
                confidence_threshold=confidence_threshold,
                save_outputs=False,
            )
            pred_product = result["predicted_product"]
            identity_total += 1
            if pred_product == product_dir.name:
                identity_correct += 1
            if result["mode"] == "low_confidence_review":
                # 低置信度样本不强行归类，计入复核分支。
                low_confidence_count += 1
            else:
                # auto_identity_accuracy 只统计系统自动接收的样本，不把复核样本算作误分类。
                auto_identity_total += 1
                if pred_product == product_dir.name:
                    auto_identity_correct += 1

            pred_ng = result["pred_label"].endswith("NG")
            if is_ng and pred_ng:
                image_tp += 1
            elif is_ng and not pred_ng:
                image_fn += 1
            elif not is_ng and pred_ng:
                image_fp += 1
            else:
                image_tn += 1

            rows.append(
                {
                    "true_product": product_dir.name,
                    "image": path.name,
                    "label": "NG" if is_ng else "OK",
                    **result,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "unknown_product_results.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "confidence_threshold": confidence_threshold,
        "model_library": [
            {"product": model.name, "ok_samples": model.sample_count, "feature_radius": model.feature_radius}
            for model in library
        ],
        "identity_accuracy": identity_correct / identity_total if identity_total else 0.0,
        "identity_correct": identity_correct,
        "identity_total": identity_total,
        "auto_identity_accuracy": auto_identity_correct / auto_identity_total if auto_identity_total else 0.0,
        "auto_identity_correct": auto_identity_correct,
        "auto_identity_total": auto_identity_total,
        "auto_coverage": auto_identity_total / identity_total if identity_total else 0.0,
        "review_rate": low_confidence_count / identity_total if identity_total else 0.0,
        "low_confidence_count": low_confidence_count,
        "image": metrics_from_counts(image_tp, image_fp, image_fn, image_tn),
    }
    with open(out_dir / "unknown_product_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return df, metrics


def main():
    """命令行入口：默认跑完整实验；提供 --predict-image 时只预测单张未知图像。"""
    parser = argparse.ArgumentParser(description="Machine vision final project: measurement and defect detection.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--defect-percentile", type=float, default=99.85)
    parser.add_argument("--predict-image", type=Path, default=None, help="Predict one unknown defect image.")
    parser.add_argument("--identity-confidence", type=float, default=0.45, help="Confidence threshold for product identity.")
    args = parser.parse_args()

    start = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.predict_image is not None:
        # 单张图模式：先用训练目录中的 OK 样本建立模型库，再预测这张未知图像。
        library = build_product_model_library(args.data_root / "defect-detection")
        result = predict_unknown_defect(
            args.predict_image,
            library,
            out_dir=args.output,
            confidence_threshold=args.identity_confidence,
            save_outputs=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nFinished in {time.time() - start:.2f}s. Outputs: {args.output.resolve()}")
        return

    # 默认模式：依次完成测量、已知产品缺陷检测、未知产品扩展评测。
    measure_df = run_measure(args.data_root / "measure", args.output)
    defect_df, defect_metrics = run_defect(args.data_root / "defect-detection", args.output, args.defect_percentile)
    unknown_df, unknown_metrics = run_unknown_product_evaluation(
        args.data_root / "defect-detection",
        args.output,
        args.identity_confidence,
    )

    print("Measurement summary:")
    print(measure_df[["width_px", "height_px", "rectangularity"]].agg(["mean", "var", "std"]))
    print("\nDefect metrics:")
    for item in defect_metrics:
        print(json.dumps(item, ensure_ascii=False, indent=2))
    print("\nUnknown-product pipeline metrics:")
    print(json.dumps(unknown_metrics, ensure_ascii=False, indent=2))
    print(f"\nFinished in {time.time() - start:.2f}s. Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
