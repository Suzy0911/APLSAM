from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter

MASK_DIR = Path("./data/masks")   # 固定的 mask 文件夹路径
MAX_NUM_SP = 7
AVG_SP_AREA = 100
DOWN_STRIDE = 16
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def place_seed_points_cpu(mask_np, down_stride=16, max_num_sp=7, avg_sp_area=100):
    """
    CPU：根据二值 mask 计算初始种子坐标（最多 max_num_sp 个），并返回 seeds 数量。
    返回 (segments, num_sp)。即使 num_sp == 1 也会返回 num_sp=1（用于均值统计）。
    """
    assert mask_np.ndim == 2, "mask must be HxW"
    mask_np = (mask_np > 0).astype(np.uint8)

    segments_x = np.zeros(max_num_sp, dtype=np.int64)
    segments_y = np.zeros(max_num_sp, dtype=np.int64)

    H, W = mask_np.shape
    down_h = int((H - 1) / down_stride + 1)
    down_w = int((W - 1) / down_stride + 1)
    m_np_down = cv2.resize(mask_np, dsize=(down_w, down_h), interpolation=cv2.INTER_NEAREST)

    nz = np.nonzero(m_np_down)
    if len(nz[0]) != 0:
        p = [np.min(nz[0]), np.min(nz[1])]
        pend = [np.max(nz[0]), np.max(nz[1])]
        m_np_roi = np.copy(m_np_down)[p[0]:pend[0] + 1, p[1]:pend[1] + 1]

        # 关键修改：有前景时，num_sp 至少为 1
        mask_area = int((m_np_roi == 1).sum())
        est = int(np.round(mask_area / float(avg_sp_area)))
        num_sp = max(1, min(est, max_num_sp))
    else:
        num_sp = 0
        m_np_roi = None
        p = [0, 0]

    # 放置 seeds：保持与原逻辑一致；当 num_sp>=2 时逐个放置。
    if (num_sp != 0) and (num_sp != 1):
        for i in range(num_sp):
            dtrans = distance_transform_edt(m_np_roi)
            dtrans = gaussian_filter(dtrans, sigma=0.1)
            coords1 = np.nonzero(dtrans == np.max(dtrans))
            segments_x[i] = coords1[0][0]
            segments_y[i] = coords1[1][0]
            m_np_roi[segments_x[i], segments_y[i]] = 0
            segments_x[i] += p[0]
            segments_y[i] += p[1]

    # 若你也想在 num_sp==1 时实际选取 1 个 seed，可取消下面注释：
    # elif num_sp == 1:
    #     dtrans = distance_transform_edt(m_np_roi)
    #     dtrans = gaussian_filter(dtrans, sigma=0.1)
    #     coords1 = np.nonzero(dtrans == np.max(dtrans))
    #     segments_x[0] = coords1[0][0] + p[0]
    #     segments_y[0] = coords1[1][0] + p[1]

    segments = np.stack([segments_x, segments_y], axis=1)  # (max_num_sp, 2)
    return segments, int(num_sp)

def load_mask_1024(path: Path) -> np.ndarray:
    """读取灰度图并确保为 1024x1024，返回二值 np.ndarray（0/1）"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if (img.shape[1], img.shape[0]) != (1024, 1024):
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    return (img > 0).astype(np.uint8)

def main():
    if not MASK_DIR.is_dir():
        raise NotADirectoryError(f"{MASK_DIR} is not a directory")

    files = sorted([p for p in MASK_DIR.iterdir() if p.suffix.lower() in EXTS])
    if not files:
        print(f"No mask files with extensions {sorted(EXTS)} found in: {MASK_DIR}")
        return

    seed_counts = []
    for p in files:
        try:
            mask = load_mask_1024(p)
            _, num_sp = place_seed_points_cpu(
                mask_np=mask,
                down_stride=DOWN_STRIDE,
                max_num_sp=MAX_NUM_SP,
                avg_sp_area=AVG_SP_AREA,
            )
            # 这里 num_sp==1 直接按 1 计入
            seed_counts.append(num_sp)
            print(f"{p.name}: init seeds = {num_sp}")
        except Exception as e:
            print(f"{p.name}: ERROR -> {e}")

    if seed_counts:
        mean_val = float(np.mean(seed_counts))
        print("-" * 40)
        print(f"Files processed: {len(seed_counts)} / {len(files)}")
        print(f"Mean init seeds across folder: {mean_val:.4f}")
    else:
        print("No valid masks were processed; mean cannot be computed.")

if __name__ == "__main__":
    main()
