import numpy as np
import SimpleITK as sitk


INPUT_SIZE = (256, 256, 256)


def normalize_01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or not arr.size or not np.isfinite(arr).all():
        raise ValueError('volume must be a finite, nonempty 3D array')
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def resample_isotropic(img, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    if img.GetDimension() != 3:
        raise ValueError('a 3D image is required')
    spacing = np.asarray(out_spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('output spacing must contain three finite positive values')
    size = [max(1, int(round((n - 1) * s / t)) + 1)
            for n, s, t in zip(img.GetSize(), img.GetSpacing(), spacing)]
    interpolation = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    pixel_type = sitk.sitkUInt16 if is_label else sitk.sitkFloat32
    return sitk.Resample(
        img, size, sitk.Transform(3, sitk.sitkIdentity), interpolation,
        img.GetOrigin(), tuple(spacing), img.GetDirection(), 0, pixel_type,
    )


def _resize_image(img, is_label=False):
    size = img.GetSize()
    spacing = tuple((n - 1) * s / (out - 1) if n > 1 else s
                    for n, s, out in zip(size, img.GetSpacing(), INPUT_SIZE))
    return sitk.Resample(
        img, INPUT_SIZE, sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        img.GetOrigin(), spacing, img.GetDirection(), 0,
        sitk.sitkUInt16 if is_label else sitk.sitkFloat32,
    )


def resize_to_256(arr, order=1):
    arr = np.asarray(arr)
    if arr.ndim != 3 or not arr.size or order not in (0, 1):
        raise ValueError('resize requires a nonempty 3D array and order 0 or 1')
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    return sitk.GetArrayFromImage(_resize_image(img, is_label=(order == 0)))


def _checked_side(side):
    side = str(side).upper() if side is not None else ''
    if side not in ('L', 'R'):
        raise ValueError("raw CT preprocessing requires an explicit side='L' or side='R'")
    return side


def _checked_crop_box(crop_box, shape):
    box = np.asarray(crop_box, dtype=float)
    if box.shape != (6,) or not np.isfinite(box).all() or np.any(box != np.floor(box)):
        raise ValueError('crop_box must contain six integer bounds in z0,z1,y0,y1,x0,x1 order')
    box = box.astype(np.int64)
    if np.any(box[::2] < 0) or np.any(box[1::2] > np.asarray(shape)) or np.any(box[1::2] <= box[::2]):
        raise ValueError('crop_box is empty or outside the oriented isotropic CT grid')
    return tuple(int(v) for v in box)


def _mask_crop_box(seg, pad=8):
    indices = np.argwhere(seg > 0)
    if not len(indices):
        raise ValueError('segmentation is empty on the CT reference grid')
    x0 = max(0, int(indices[:, 2].min()) - pad)
    x1 = min(seg.shape[2], int(indices[:, 2].max()) + pad + 1)
    return (0, seg.shape[0], 0, seg.shape[1], x0, x1)


def crop_unilateral(arr, seg, pad=8):
    arr, seg = np.asarray(arr), np.asarray(seg)
    if arr.ndim != 3 or arr.shape != seg.shape:
        raise ValueError('CT and segmentation must share the same 3D grid')
    z0, z1, y0, y1, x0, x1 = _mask_crop_box(seg, pad)
    return arr[z0:z1, y0:y1, x0:x1]


def flip_right_to_left(arr, side):
    return np.ascontiguousarray(arr[:, :, ::-1] if _checked_side(side) == 'R' else arr)


def apply_412(ct_path, seg_path=None, *, side=None, crop_box=None, return_seg=False):
    side = _checked_side(side)
    image = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    if image.GetDimension() != 3:
        raise ValueError('a 3D CT image is required')
    image = resample_isotropic(sitk.DICOMOrient(image, 'LPS'))
    segmentation = None
    if seg_path is not None:
        source = sitk.ReadImage(str(seg_path))
        if source.GetDimension() != 3:
            raise ValueError('a 3D segmentation is required')
        segmentation = sitk.Resample(
            source, image, sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
        )
    shape = image.GetSize()[::-1]
    if crop_box is None:
        if segmentation is None:
            raise ValueError('raw CT without segmentation requires crop_box; use raw=False for preprocessed inputs')
        crop_box = _mask_crop_box(sitk.GetArrayFromImage(segmentation))
    z0, z1, y0, y1, x0, x1 = _checked_crop_box(crop_box, shape)
    size, index = [x1 - x0, y1 - y0, z1 - z0], [x0, y0, z0]
    image = sitk.RegionOfInterest(image, size, index)
    if segmentation is not None:
        segmentation = sitk.RegionOfInterest(segmentation, size, index)
    if side == 'R':
        image = sitk.Flip(image, [True, False, False])
        if segmentation is not None:
            segmentation = sitk.Flip(segmentation, [True, False, False])
    image = _resize_image(image)
    volume = normalize_01(sitk.GetArrayFromImage(image))
    mask = None
    if segmentation is not None:
        segmentation = sitk.Resample(
            segmentation, image, sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
        )
        mask = np.ascontiguousarray(sitk.GetArrayFromImage(segmentation))
    return (volume, mask) if return_seg else volume


def load_preprocessed(nii_path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nii_path)))
    if arr.shape != INPUT_SIZE:
        raise ValueError(f'preprocessed volume must have shape {INPUT_SIZE}, got {arr.shape}')
    return normalize_01(arr)


def load_preprocessed_segmentation(ct_path, seg_path):
    image, segmentation = sitk.ReadImage(str(ct_path)), sitk.ReadImage(str(seg_path))
    if image.GetDimension() != 3 or segmentation.GetDimension() != 3:
        raise ValueError('CT and segmentation must be 3D')
    matches = image.GetSize() == segmentation.GetSize() and all(
        np.allclose(left, right, rtol=0, atol=1e-5)
        for left, right in ((image.GetOrigin(), segmentation.GetOrigin()),
                            (image.GetSpacing(), segmentation.GetSpacing()),
                            (image.GetDirection(), segmentation.GetDirection()))
    )
    if image.GetSize() != INPUT_SIZE or not matches:
        raise ValueError('raw=False requires segmentation preprocessed onto the same 256-cubed CT grid')
    return np.ascontiguousarray(sitk.GetArrayFromImage(segmentation))


def structural(seg):
    from skimage import measure
    mask = (np.asarray(seg) > 0).astype(np.float32)
    if mask.ndim != 3:
        raise ValueError('structural prior requires a 3D mask')
    boundary = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        for contour in measure.find_contours(mask[z], level=0.5):
            points = np.round(contour).astype(int)
            keep = ((points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) &
                    (points[:, 1] >= 0) & (points[:, 1] < mask.shape[2]))
            points = points[keep]
            boundary[z, points[:, 0], points[:, 1]] = 3.0
    return np.where(boundary == 3.0, 3.0, mask).astype(np.float32)
