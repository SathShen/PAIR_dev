# -*- coding: utf-8 -*-
"""
PAIR standard dataset interface.

Directory is schema. The existing 2D SCD/BCD protocols are unchanged.

Canonical 3D point files:
    points_t*/sample.npz
        coord      [N,3] mandatory
        rgb        [N,3] optional
        intensity  [N,1] optional

Missing optional attributes are represented by field absence, never fabricated
zeros. RGB/intensity are preserved as float32 without per-sample normalization;
normalization belongs to the model/dataset-specific preparation policy.

Canonical 3D supervision files:
    semantic_t*/sample.npz
        semantic     [N]
        event        [N]
        event_valid  [N] optional bool

PAIR 3D event taxonomy:
    0 unchanged
    1 added
    2 removed
    3 class_change
    4 height_up
    5 height_down

T1 and T2 are independent point topologies; N1 need not equal N2. Point and
supervision files for one epoch must have identical point ordering.

Semantic ignore policy:
    DatasetSpec.ignored_id is optional and defaults to None. There is no magic
    ignore value. Every observed semantic ID must be declared in class_names or
    equal the explicitly configured ignored_id; otherwise loading raises.

Pure 3D training keeps the existing 51.2m shared XY crop policy. Oversized
validation/test scenes still require deterministic tiling.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
VALID_MODALITIES = {'image', 'point'}
PAIR_POINT_WINDOW_SIZE_M = 51.2
PAIR_EVENT_NAMES = {0: 'unchanged', 1: 'added', 2: 'removed', 3: 'class_change', 4: 'height_up', 5: 'height_down'}
PAIR_EVENT_NUM_CLASSES = len(PAIR_EVENT_NAMES)

def normalize_class_name(name: str) -> str:
    return ' '.join(str(name).strip().lower().replace('_', ' ').replace('-', ' ').split())

def infer_unchanged_raw_id(class_names: Dict[int, str]) -> Optional[int]:
    """
    Infer an explicit semantic 'unchanged' class by NAME only.
    Intentionally does NOT treat 'background' as unchanged.
    """
    aliases = {'unchanged', 'no change', 'non change'}
    matches = [int(raw_id) for raw_id, name in class_names.items() if normalize_class_name(name) in aliases]
    if len(matches) > 1:
        raise ValueError(f'Multiple unchanged-like classes found: {matches}. Use one explicit unchanged/no-change class name.')
    return matches[0] if matches else None

def infer_binary_class_ids(class_names: Dict[int, str]) -> Tuple[int, int]:
    """
    Infer physical unchanged/changed raw IDs from class_names.

    No changed_raw_id field is stored in DatasetSpec.

    Examples:
        {0: "unchanged", 255: "changed"} -> (0, 255)
        {0: "unchanged", 1: "changed"}   -> (0, 1)
    """
    unchanged_aliases = {'unchanged', 'no change', 'non change'}
    changed_aliases = {'changed', 'change'}
    unchanged = [int(raw_id) for raw_id, name in class_names.items() if normalize_class_name(name) in unchanged_aliases]
    changed = [int(raw_id) for raw_id, name in class_names.items() if normalize_class_name(name) in changed_aliases]
    if len(unchanged) != 1:
        raise ValueError(f'Binary class_names must contain exactly one unchanged/no-change class; found raw IDs {unchanged}.')
    if len(changed) != 1:
        raise ValueError(f'Binary class_names must contain exactly one changed/change class; found raw IDs {changed}.')
    if unchanged[0] == changed[0]:
        raise ValueError('Binary unchanged and changed raw IDs must differ.')
    return (unchanged[0], changed[0])

def route_from_modalities(modalities: Sequence[str]) -> str:
    m = set(modalities)
    if m == {'image'}:
        return '2d'
    if m == {'point'}:
        return '3d'
    if m == {'image', 'point'}:
        return '2d3d'
    raise ValueError(f'Unsupported modality set: {sorted(m)}')

@dataclass(frozen=True)
class DatasetSpec:
    """
    Internal normalized dataset description.

    Users do not create this in config. config_loader builds it automatically
    from the prepared PAIR directory and the dataset's class_names.
    """
    name: str
    modalities: Tuple[str, ...]
    label_mode: str
    class_names: Dict[int, str]
    ignored_id: Optional[int] = None

    @property
    def route(self) -> str:
        return route_from_modalities(self.modalities)

    @property
    def has_image(self) -> bool:
        return 'image' in self.modalities

    @property
    def has_point(self) -> bool:
        return 'point' in self.modalities

@dataclass
class CanonicalChangeTarget:
    """
    Common-topology target used by the existing 2D path.

    Valid masks are optional. For a fully supervised semantic-pair dataset with
    ignored_id=None, they stay None and are not materialized. Masks are only
    created when they carry real information, e.g. an explicit ignored class or
    intentionally missing semantic supervision in binary/post-semantic modes.
    """
    change: torch.Tensor
    semantic_t1: torch.Tensor
    semantic_t2: torch.Tensor
    change_valid: Optional[torch.Tensor] = None
    semantic_valid_t1: Optional[torch.Tensor] = None
    semantic_valid_t2: Optional[torch.Tensor] = None

def _long(x):
    if torch.is_tensor(x):
        return x.long()
    return torch.as_tensor(np.asarray(x), dtype=torch.long)

def _declared_class_ids(class_names: Dict[int, str]) -> set[int]:
    if not isinstance(class_names, dict) or not class_names:
        raise TypeError('class_names must be a non-empty Dict[int, str]')
    return {int(raw_id) for raw_id in class_names.keys()}

def validate_ignore_config(class_names: Dict[int, str], ignored_id: Optional[int]):
    """
    class_names contains trainable semantic classes only. If a source dataset
    has one explicit ignored class, it is declared separately by ignored_id.
    """
    if ignored_id is None:
        return
    ignored_id = int(ignored_id)
    if ignored_id in _declared_class_ids(class_names):
        raise ValueError(f'ignored_id={ignored_id} is also present in class_names. Ignored classes must not occupy a trainable classifier slot.')

def validate_semantic_label_values(raw: torch.Tensor, class_names: Dict[int, str], *, ignored_id: Optional[int]=None, label_name: str='semantic') -> Optional[torch.Tensor]:
    """
    Validate a raw semantic target against the dataset protocol.

    Returns:
        None when ignored_id is None (all elements are valid), otherwise a
        bool mask where True means the semantic element participates in loss.

    Unknown IDs are never converted or silently ignored.
    """
    raw = _long(raw)
    declared = _declared_class_ids(class_names)
    validate_ignore_config(class_names, ignored_id)
    allowed = torch.zeros_like(raw, dtype=torch.bool)
    for raw_id in declared:
        allowed |= raw == raw_id
    if ignored_id is not None:
        ignored_id = int(ignored_id)
        allowed |= raw == ignored_id
    if not allowed.all():
        bad = torch.unique(raw[~allowed]).cpu().tolist()
        allowed_ids = sorted(declared)
        suffix = f' plus ignored_id={int(ignored_id)}' if ignored_id is not None else ''
        raise ValueError(f'{label_name} contains undeclared raw IDs {bad}. class_names declares {allowed_ids}{suffix}.')
    if ignored_id is None:
        return None
    return raw != int(ignored_id)

def _validate_canonical_change_values(raw: torch.Tensor):
    """
    Explicit PAIR change supervision is strictly binary.

    There is no implicit -100/void value. If a future change-only dataset has a
    physical ignore code, it must be represented explicitly by that dataset's
    protocol rather than being silently accepted here.
    """
    valid_values = (raw == 0) | (raw == 1)
    if not valid_values.all():
        bad = torch.unique(raw[~valid_values]).cpu().tolist()
        raise ValueError(f'Canonical PAIR change targets must contain only 0=unchanged and 1=changed; found {bad}')

def _normalize_binary_change_values(raw: torch.Tensor, class_names: Dict[int, str], *, ignored_id: Optional[int]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert physical binary labels to PAIR internal 0/1 using class_names.

    An optional ignored_id is allowed for binary datasets as an explicit
    source value. No other undeclared value is accepted.
    """
    unchanged_raw_id, changed_raw_id = infer_binary_class_ids(class_names)
    if ignored_id is not None:
        ignored_id = int(ignored_id)
        if ignored_id in {unchanged_raw_id, changed_raw_id}:
            raise ValueError(f'ignored_id={ignored_id} conflicts with the binary unchanged/changed class IDs')
    valid_values = (raw == unchanged_raw_id) | (raw == changed_raw_id)
    if ignored_id is not None:
        valid_values |= raw == ignored_id
    if not valid_values.all():
        bad = torch.unique(raw[~valid_values]).cpu().tolist()
        raise ValueError(f'Binary change target contains undeclared raw IDs {bad}; class_names declares unchanged={unchanged_raw_id}, changed={changed_raw_id}' + (f', ignored_id={ignored_id}.' if ignored_id is not None else '.'))
    out = torch.zeros_like(raw)
    out[raw == changed_raw_id] = 1
    change_valid = None
    if ignored_id is not None:
        change_valid = raw != ignored_id
    return (out, change_valid)

def build_canonical_target(*, label_mode: str, semantic_t1=None, semantic_t2=None, change=None, class_names: Optional[Dict[int, str]]=None, ignored_id: Optional[int]=None) -> CanonicalChangeTarget:
    """
    Build a common-topology target for the existing 2D path.

    Strict policy:
      * semantic IDs must be declared in class_names;
      * one explicitly configured ignored_id is additionally allowed;
      * any other semantic ID raises immediately;
      * canonical explicit change supervision must be 0/1 only;
      * when ignored_id is None, normal semantic-pair data do not allocate
        redundant all-True valid masks.

    Ragged 3D point pairs use build_bitemporal_point_target() instead.
    """
    mode = str(label_mode).lower().strip()
    if mode not in {'semantic_pair', 'post_semantic', 'binary'}:
        raise ValueError(f'Unsupported label_mode: {label_mode}')
    if mode == 'semantic_pair':
        if semantic_t1 is None or semantic_t2 is None:
            raise ValueError('semantic_pair requires semantic_t1 and semantic_t2')
        if class_names is None:
            raise ValueError('semantic_pair validation requires class_names')
        s1 = _long(semantic_t1)
        s2 = _long(semantic_t2)
        if s1.shape != s2.shape:
            raise ValueError(f'Common-topology semantic_pair requires matching shapes; got semantic_t1={tuple(s1.shape)} and semantic_t2={tuple(s2.shape)}. Ragged 3D point pairs must use per-epoch supervision bundles.')
        sem_valid_t1 = validate_semantic_label_values(s1, class_names, ignored_id=ignored_id, label_name='semantic_t1')
        sem_valid_t2 = validate_semantic_label_values(s2, class_names, ignored_id=ignored_id, label_name='semantic_t2')
        if change is None:
            ch = (s1 != s2).long()
        else:
            ch = _long(change)
            if ch.shape != s1.shape:
                raise ValueError('change mask must match semantic label shape')
            _validate_canonical_change_values(ch)
        change_valid = None
        if ignored_id is not None:
            change_valid = sem_valid_t1 & sem_valid_t2
        return CanonicalChangeTarget(change=ch, semantic_t1=s1, semantic_t2=s2, 
                                     change_valid=change_valid, semantic_valid_t1=sem_valid_t1, semantic_valid_t2=sem_valid_t2)
    if change is None:
        raise ValueError(f'{mode} requires change supervision')
    raw = _long(change)
    if mode == 'binary':
        if class_names is None:
            raise ValueError('binary target normalization requires class_names')
        ch, change_valid = _normalize_binary_change_values(raw, class_names, ignored_id=ignored_id)
        s1 = torch.zeros_like(raw)
        s2 = torch.zeros_like(raw)
        sem_valid = torch.zeros_like(raw, dtype=torch.bool)
        return CanonicalChangeTarget(change=ch, semantic_t1=s1, semantic_t2=s2, change_valid=change_valid, 
                                     semantic_valid_t1=sem_valid, semantic_valid_t2=sem_valid.clone())
    _validate_canonical_change_values(raw)
    ch = raw.clone()
    if semantic_t2 is None:
        raise ValueError('post_semantic requires semantic_t2')
    if class_names is None:
        raise ValueError('post_semantic validation requires class_names')
    s2 = _long(semantic_t2)
    if s2.shape != raw.shape:
        raise ValueError('semantic_t2 must match change mask shape')
    sem_valid_t2 = validate_semantic_label_values(s2, class_names, ignored_id=ignored_id, label_name='semantic_t2')
    s1 = torch.zeros_like(raw)
    return CanonicalChangeTarget(change=ch, semantic_t1=s1, semantic_t2=s2, change_valid=None, 
                                 semantic_valid_t1=torch.zeros_like(raw, dtype=torch.bool), semantic_valid_t2=sem_valid_t2)

def build_default_prompt(spec: DatasetSpec) -> str:
    if spec.has_point:
        text = 'Perform 3D semantic change reasoning between Time 1 and Time 2. Predict the semantic class at both times and the per-point change event.'
    else:
        text = 'Perform semantic change detection between Time 1 and Time 2. Identify unchanged and changed regions.'
        if spec.label_mode == 'semantic_pair':
            text += ' For changed regions, infer the semantic class before and after change.'
        elif spec.label_mode == 'post_semantic':
            text += ' The pre-change semantic class may be unknown, while the post-change class is supervised.'
        elif spec.label_mode == 'binary':
            text += ' The source dataset supervises change only; semantic classes before and after change may be unknown.'
    if spec.class_names:
        classes = ', '.join((f'{raw_id}: {name}' for raw_id, name in spec.class_names.items()))
        text += ' Valid semantic classes are: ' + classes + '.'
    return text

def read_label_array(path: Union[str, Path]) -> torch.Tensor:
    """
    Read one single-topology label array.

    Used by the existing 2D pipeline. A multi-array NPZ is intentionally
    rejected here; 3D semantic/event bundles use read_point_supervision().
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path, allow_pickle=False)
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
            if len(keys) != 1:
                raise ValueError(f'{path} has multiple arrays {keys}; use read_point_supervision() for 3D supervision bundles')
            arr = np.asarray(z[keys[0]])
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f'Expected a single-channel label, got {arr.shape}: {path}')
    return torch.as_tensor(np.array(arr, copy=True), dtype=torch.long)

def _validate_event_values(event: torch.Tensor, label_name: str='event'):
    event = _long(event)
    valid = (event >= 0) & (event < PAIR_EVENT_NUM_CLASSES)
    if not valid.all():
        bad = torch.unique(event[~valid]).cpu().tolist()
        raise ValueError(f'{label_name} contains invalid event IDs {bad}; expected 0..{PAIR_EVENT_NUM_CLASSES - 1}')

def read_point_supervision(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Read one epoch's prepared 3D semantic/event supervision bundle."""
    path = Path(path)
    if path.suffix.lower() != '.npz':
        raise ValueError(f'PAIR 3D supervision must be NPZ: {path}')
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.keys())
        missing = sorted({'semantic', 'event'} - keys)
        if missing:
            raise ValueError(f'{path} is missing supervision arrays {missing}; available={sorted(keys)}')
        semantic = np.asarray(z['semantic'])
        event = np.asarray(z['event'])
        event_valid = np.asarray(z['event_valid']) if 'event_valid' in z else None
    if semantic.ndim != 1 or event.ndim != 1:
        raise ValueError(f'{path}: semantic/event must both be 1D [N]')
    if semantic.shape != event.shape:
        raise ValueError(f'{path}: semantic/event shapes differ: {semantic.shape} vs {event.shape}')
    if event_valid is not None and event_valid.shape != semantic.shape:
        raise ValueError(f'{path}: event_valid shape {event_valid.shape} does not match semantic {semantic.shape}')
    out = {'semantic': torch.as_tensor(np.array(semantic, copy=True), dtype=torch.long), 
           'event': torch.as_tensor(np.array(event, copy=True), dtype=torch.long)}
    _validate_event_values(out['event'], f'{path} event')
    if event_valid is not None:
        out['event_valid'] = torch.as_tensor(np.array(event_valid, copy=True), dtype=torch.bool)
    return out

def _read_geotiff(path: Path):
    if path.suffix.lower() not in {'.tif', '.tiff'}:
        return None
    try:
        import rasterio
    except ImportError:
        return None
    with rasterio.open(path) as src:
        arr = src.read()
        meta = {'crs': str(src.crs) if src.crs is not None else None, 'transform': tuple(src.transform), 'bounds': tuple(src.bounds), 'width': int(src.width), 'height': int(src.height), 'gsd_x': abs(float(src.transform.a)), 'gsd_y': abs(float(src.transform.e))}
    return (np.moveaxis(arr, 0, -1), meta)

def read_image(path: Union[str, Path]):
    path = Path(path)
    geo = _read_geotiff(path)
    if geo is not None:
        arr, meta = geo
    else:
        im = Image.open(path).convert('RGB')
        arr = np.asarray(im)
        meta = {'crs': None, 'transform': None, 'bounds': None, 'width': im.width, 'height': im.height, 'gsd_x': None, 'gsd_y': None}
    if arr.ndim == 2:
        arr = arr[..., None]
    if np.issubdtype(arr.dtype, np.integer):
        denom = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / denom
    else:
        arr = arr.astype(np.float32)
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if vmax > 1.5:
            arr /= 255.0 if vmax <= 255.0 else 65535.0
    tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(arr, -1, 0))).float()
    return (tensor, meta)

def same_geo_grid(meta1: Dict[str, Any], meta2: Dict[str, Any], atol=1e-07) -> bool:
    crs1 = meta1.get('crs')
    crs2 = meta2.get('crs')
    tr1 = meta1.get('transform')
    tr2 = meta2.get('transform')
    if crs1 is None and crs2 is None and (tr1 is None) and (tr2 is None):
        return True
    if crs1 != crs2 or tr1 is None or tr2 is None:
        return False
    return np.allclose(np.asarray(tr1), np.asarray(tr2), atol=atol, rtol=0.0) and meta1['width'] == meta2['width'] and (meta1['height'] == meta2['height'])
TargetBuilder = Callable[[Dict[str, Any], DatasetSpec], Dict[str, Any]]

def _read_las(path: Path):
    try:
        import laspy
    except ImportError as exc:
        raise ImportError('LAS/LAZ reading requires laspy') from exc
    las = laspy.read(path)
    attrs: Dict[str, Any] = {'coord': np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1).astype(np.float32)}
    available = set(las.point_format.dimension_names)
    for name in ('intensity', 'red', 'green', 'blue', 'nir', 'classification', 'return_number', 'number_of_returns'):
        if name in available:
            attrs[name] = np.asarray(getattr(las, name))
    try:
        crs = las.header.parse_crs()
        attrs['_crs'] = None if crs is None else str(crs)
    except Exception:
        attrs['_crs'] = None
    return attrs

def _optional_point_tensor(array, *, n: int, width: int, name: str, path: Path):
    arr = np.asarray(array)
    if width == 1 and arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape != (n, width):
        raise ValueError(f'{path}: {name} must be [{n},{width}], got {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError(f'{path}: {name} contains NaN/Inf')
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32, copy=False))).float()

def read_point_cloud(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read PAIR point input: coord mandatory, rgb/intensity optional.

    Optional fields are never synthesized. Values are cast to float32 but not
    normalized here. Canonical PAIR point files no longer use a generic feat field.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    crs = None
    rgb = None
    intensity = None
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.keys())
            if 'coord' not in keys:
                raise ValueError(f'{path} must contain coord [N,3]')
            coord = np.asarray(z['coord'], dtype=np.float32)
            if 'feat' in keys:
                raise ValueError(f"{path} contains legacy 'feat'. Re-prepare using coord + optional rgb/intensity.")
            if 'rgb' in keys:
                rgb = np.asarray(z['rgb'])
            if 'intensity' in keys:
                intensity = np.asarray(z['intensity'])
            if 'crs' in keys:
                value = np.asarray(z['crs'])
                crs = str(value.item() if value.ndim == 0 else value)
    elif suffix == '.npy':
        coord = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f'{path}: canonical NPY point input must be [N,3]')
    elif suffix in {'.las', '.laz'}:
        attrs = _read_las(path)
        coord = attrs['coord']
        crs = attrs.get('_crs')
        if all((name in attrs for name in ('red', 'green', 'blue'))):
            rgb = np.column_stack((attrs['red'], attrs['green'], attrs['blue']))
        if 'intensity' in attrs:
            intensity = np.asarray(attrs['intensity'])
    else:
        raise ValueError(f'Unsupported point-cloud format: {path}')
    if coord.ndim != 2 or coord.shape[1] != 3 or coord.shape[0] == 0:
        raise ValueError(f'{path}: coord must be non-empty [N,3], got {coord.shape}')
    if not np.isfinite(coord).all():
        raise ValueError(f'{path}: coord contains NaN/Inf')
    n = coord.shape[0]
    out: Dict[str, Any] = {'coord': torch.from_numpy(np.ascontiguousarray(coord)).float(), 'crs': crs}
    if rgb is not None:
        out['rgb'] = _optional_point_tensor(rgb, n=n, width=3, name='rgb', path=path)
    if intensity is not None:
        out['intensity'] = _optional_point_tensor(intensity, n=n, width=1, name='intensity', path=path)
    return out

def point_xy_bounds(p1: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Union XY bounds of a temporal point pair.
    """
    if p1['coord'].shape[0] == 0 or p2['coord'].shape[0] == 0:
        raise ValueError('Cannot compute bounds of an empty temporal point pair')
    xmin = min(float(p1['coord'][:, 0].min().item()), float(p2['coord'][:, 0].min().item()))
    ymin = min(float(p1['coord'][:, 1].min().item()), float(p2['coord'][:, 1].min().item()))
    xmax = max(float(p1['coord'][:, 0].max().item()), float(p2['coord'][:, 0].max().item()))
    ymax = max(float(p1['coord'][:, 1].max().item()), float(p2['coord'][:, 1].max().item()))
    return (xmin, ymin, xmax, ymax)

def _window_start(low: float, high: float, size: float, *, randomize: bool) -> float:
    """
    Choose the start coordinate of a fixed physical window.

    If source extent is smaller than the window, center the source inside the
    canonical window. This keeps the returned crop physically size-consistent.
    """
    extent = high - low
    if extent <= size:
        return 0.5 * (low + high) - 0.5 * size
    if not randomize:
        return 0.5 * (low + high - size)
    u = float(torch.rand((), dtype=torch.float64).item())
    return low + u * (extent - size)

def make_random_pair_window(p1: Dict[str, Any], p2: Dict[str, Any], *, size: float=PAIR_POINT_WINDOW_SIZE_M) -> Tuple[float, float, float, float]:
    """
    Sample ONE XY window and apply it to both epochs.

    The returned window is always size × size in world coordinates.
    """
    if size <= 0:
        raise ValueError('Point spatial window size must be > 0')
    xmin, ymin, xmax, ymax = point_xy_bounds(p1, p2)
    x0 = _window_start(xmin, xmax, size, randomize=True)
    y0 = _window_start(ymin, ymax, size, randomize=True)
    return (x0, y0, x0 + size, y0 + size)

def crop_points_xy(point_data: Dict[str, Any], bounds):
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f'Invalid XY bounds: {bounds}')
    coord = point_data['coord']
    keep = (coord[:, 0] >= xmin) & (coord[:, 0] < xmax) & (coord[:, 1] >= ymin) & (coord[:, 1] < ymax)
    out = {**point_data, 'coord': coord[keep]}
    for key in ('rgb', 'intensity'):
        if key in point_data:
            out[key] = point_data[key][keep]
    out['_crop_mask'] = keep
    out['_source_indices'] = torch.nonzero(keep, as_tuple=False).flatten()
    return out

def crop_point_supervision(target: Dict[str, torch.Tensor], keep: torch.Tensor) -> Dict[str, torch.Tensor]:
    if keep.dtype != torch.bool or keep.ndim != 1:
        raise ValueError('Point crop mask must be bool [N]')
    out = {}
    for key in ('semantic', 'event', 'event_valid'):
        if key not in target:
            continue
        value = target[key]
        if value.ndim != 1 or value.shape[0] != keep.shape[0]:
            raise ValueError(f'Cannot crop {key}: target shape={tuple(value.shape)}, mask={tuple(keep.shape)}')
        out[key] = value[keep]
    return out

def build_bitemporal_point_target(supervision_t1: Dict[str, torch.Tensor], supervision_t2: Dict[str, torch.Tensor], *, class_names: Dict[int, str], ignored_id: Optional[int]=None) -> Dict[str, torch.Tensor]:
    """Build ragged T1/T2 semantic + 6-way event targets without index correspondence."""
    s1 = _long(supervision_t1['semantic'])
    s2 = _long(supervision_t2['semantic'])
    e1 = _long(supervision_t1['event'])
    e2 = _long(supervision_t2['event'])
    if s1.ndim != 1 or e1.ndim != 1 or s1.shape != e1.shape:
        raise ValueError('3D T1 semantic/event must be same-length 1D [N1]')
    if s2.ndim != 1 or e2.ndim != 1 or s2.shape != e2.shape:
        raise ValueError('3D T2 semantic/event must be same-length 1D [N2]')
    sem_valid_t1 = validate_semantic_label_values(s1, class_names, ignored_id=ignored_id, label_name='3D semantic_t1')
    sem_valid_t2 = validate_semantic_label_values(s2, class_names, ignored_id=ignored_id, label_name='3D semantic_t2')
    _validate_event_values(e1, '3D event_t1')
    _validate_event_values(e2, '3D event_t2')
    target: Dict[str, torch.Tensor] = {'semantic_t1': s1, 'semantic_t2': s2, 'event_t1': e1, 'event_t2': e2}
    src_event_valid_t1 = supervision_t1.get('event_valid')
    src_event_valid_t2 = supervision_t2.get('event_valid')
    if src_event_valid_t1 is not None:
        src_event_valid_t1 = src_event_valid_t1.bool()
    if src_event_valid_t2 is not None:
        src_event_valid_t2 = src_event_valid_t2.bool()
    if sem_valid_t1 is not None:
        target['semantic_valid_t1'] = sem_valid_t1
    if sem_valid_t2 is not None:
        target['semantic_valid_t2'] = sem_valid_t2
    event_valid_t1 = src_event_valid_t1
    event_valid_t2 = src_event_valid_t2
    if sem_valid_t1 is not None:
        event_valid_t1 = sem_valid_t1 if event_valid_t1 is None else event_valid_t1 & sem_valid_t1
    if sem_valid_t2 is not None:
        event_valid_t2 = sem_valid_t2 if event_valid_t2 is None else event_valid_t2 & sem_valid_t2
    if event_valid_t1 is not None:
        target['event_valid_t1'] = event_valid_t1
    if event_valid_t2 is not None:
        target['event_valid_t2'] = event_valid_t2
    return target

def make_point_dict(point_data: Dict[str, Any], *, shared_xyz_origin=None) -> Dict[str, torch.Tensor]:
    """Translate one epoch to the shared local XYZ frame and preserve optional attributes."""
    coord = point_data['coord'].clone()
    if shared_xyz_origin is None:
        shared_xyz_origin = tuple((float(v) for v in coord.amin(dim=0).tolist()))
    origin = torch.tensor(shared_xyz_origin, dtype=coord.dtype, device=coord.device).view(1, 3)
    out: Dict[str, torch.Tensor] = {'coord': coord - origin}
    for key in ('rgb', 'intensity'):
        if key in point_data:
            out[key] = point_data[key]
    return out

def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{i}') from exc
    return records

def resolve_dataset_path(path: Union[str, Path], dataset_root: Optional[Union[str, Path]]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if dataset_root is None:
        raise ValueError(f'Relative path {path} requires dataset_root')
    return Path(dataset_root).expanduser().resolve() / path

def _infer_split_from_records(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    """
    Fallback split inference when records were supplied directly rather than
    through manifests/train.jsonl etc.
    """
    found = set()
    for record in records:
        sample_id = str(record.get('id', '')).lower()
        for split in ('train', 'val', 'test'):
            if sample_id.startswith(split + '_'):
                found.add(split)
                break
    if len(found) == 1:
        return next(iter(found))
    return None

class UnifiedPAIRDataset(Dataset):
    """
    Manifest-driven reader for an already prepared PAIR-standard dataset.

    Directory layout decides route and supervision. Manifest records map
    sample IDs to concrete files.

    3D training spatial policy
    --------------------------
    - source union extent <= 51.2 m in both X/Y:
        use the complete temporal pair.
    - larger pure-3D train source:
        select one random shared 51.2 × 51.2 m XY window.
    - val/test:
        never random-crop. Oversized evaluation scenes must be handled by a
        deterministic tiling adapter rather than silently evaluated partially.

    2D+3D:
        point bounds must come from the common image footprint or an explicit
        manifest bounds field. Points are never independently randomly cropped
        away from the image footprint.
    """

    def __init__(self, records: Union[Sequence[Dict[str, Any]], str, Path], spec: DatasetSpec, *, target_builder: Optional[TargetBuilder]=None):
        super().__init__()
        self.manifest_path = None
        self.dataset_root = None
        if isinstance(records, (str, Path)):
            self.manifest_path = Path(records).expanduser().resolve()
            self.dataset_root = self.manifest_path.parent.parent
            self.records = load_jsonl(self.manifest_path)
            manifest_split = self.manifest_path.stem.lower()
            self.split = manifest_split if manifest_split in {'train', 'val', 'test'} else None
        else:
            self.records = list(records)
            self.split = _infer_split_from_records(self.records)
        self.spec = spec
        validate_ignore_config(self.spec.class_names, self.spec.ignored_id)
        self.route = spec.route
        self.target_builder = target_builder
        self.prompt = build_default_prompt(spec)
        if not self.records:
            raise ValueError(f'Dataset {spec.name} contains no records: {self.manifest_path}')
        if self.spec.has_point and self.target_builder is not None:
            raise NotImplementedError('Custom target_builder is not supported for point routes because point supervision must be cropped with exactly the same per-point mask as coord and optional point attributes.')

    def __len__(self):
        return len(self.records)

    def _path(self, value: Union[str, Path]) -> Path:
        return resolve_dataset_path(value, self.dataset_root)

    def _load_2d(self, record):
        im1, geo1 = read_image(self._path(record['image_t1']))
        im2, geo2 = read_image(self._path(record['image_t2']))
        if not same_geo_grid(geo1, geo2):
            raise ValueError('T1/T2 rasters are not on one geospatial grid. PAIR data preparation must align CRS/GSD/affine first.')
        if im1.shape[-2:] != im2.shape[-2:]:
            raise ValueError(f'T1/T2 image sizes differ: {tuple(im1.shape[-2:])} vs {tuple(im2.shape[-2:])}. Prepare aligned pairs before training.')
        return {'images_t1': im1, 'images_t2': im2, 'geo_t1': geo1, 'geo_t2': geo2}

    def _load_point_supervision_pair(self, record, p1, p2):
        if self.spec.label_mode != 'semantic_pair':
            raise NotImplementedError('The current point-route loader is connected first for semantic_pair supervision with per-epoch semantic/event bundles. Binary/post-semantic 3D topology is not connected.')
        if 'semantic_t1' not in record or 'semantic_t2' not in record:
            raise KeyError('3D semantic_pair manifest requires semantic_t1 and semantic_t2 paths')
        t1 = read_point_supervision(self._path(record['semantic_t1']))
        t2 = read_point_supervision(self._path(record['semantic_t2']))
        validate_semantic_label_values(t1['semantic'], self.spec.class_names, ignored_id=self.spec.ignored_id, label_name=f"{self.spec.name}/{record.get('id')} semantic_t1")
        validate_semantic_label_values(t2['semantic'], self.spec.class_names, ignored_id=self.spec.ignored_id, label_name=f"{self.spec.name}/{record.get('id')} semantic_t2")
        n1 = p1['coord'].shape[0]
        n2 = p2['coord'].shape[0]
        if t1['semantic'].shape[0] != n1:
            raise ValueError(f"{self.spec.name}/{record.get('id')}: T1 point/supervision length mismatch: points={n1}, labels={t1['semantic'].shape[0]}")
        if t2['semantic'].shape[0] != n2:
            raise ValueError(f"{self.spec.name}/{record.get('id')}: T2 point/supervision length mismatch: points={n2}, labels={t2['semantic'].shape[0]}")
        return (t1, t2)

    def _choose_point_bounds(self, record, p1, p2, *, reference_bounds=None):
        """
        Decide one common XY spatial window for both epochs.

        Priority:
        1. reference_bounds from a 2D image footprint;
        2. explicit manifest record["bounds"];
        3. automatic pure-3D policy.
        """
        if reference_bounds is not None:
            if len(reference_bounds) != 4:
                raise ValueError('reference_bounds must be [xmin,ymin,xmax,ymax]')
            return tuple((float(v) for v in reference_bounds))
        explicit = record.get('bounds')
        if explicit is not None:
            if len(explicit) != 4:
                raise ValueError('bounds must be [xmin,ymin,xmax,ymax]')
            return tuple((float(v) for v in explicit))
        if self.route == '2d3d':
            raise ValueError('2D+3D point cropping requires a world-coordinate image footprint (GeoTIFF bounds) or manifest bounds. PAIR will not independently random-crop point clouds away from their paired images.')
        xmin, ymin, xmax, ymax = point_xy_bounds(p1, p2)
        extent_x = xmax - xmin
        extent_y = ymax - ymin
        if extent_x <= PAIR_POINT_WINDOW_SIZE_M and extent_y <= PAIR_POINT_WINDOW_SIZE_M:
            return None
        if self.split == 'train':
            return make_random_pair_window(p1, p2, size=PAIR_POINT_WINDOW_SIZE_M)
        if self.split in {'val', 'test'}:
            raise NotImplementedError(f"{self.spec.name}/{record.get('id')}: evaluation source extent {extent_x:.2f}m × {extent_y:.2f}m exceeds PAIR's {PAIR_POINT_WINDOW_SIZE_M:.1f}m window. Validation/test must use deterministic tiling; random evaluation cropping is intentionally forbidden.")
        raise RuntimeError(f"{self.spec.name}/{record.get('id')}: cannot infer train/val/test split for an oversized 3D source. Use manifests/train.jsonl, val.jsonl, test.jsonl or explicit bounds.")

    def _load_3d(self, record, *, reference_bounds=None):
        p1 = read_point_cloud(self._path(record['point_t1']))
        p2 = read_point_cloud(self._path(record['point_t2']))
        if p1['coord'].shape[0] == 0 or p2['coord'].shape[0] == 0:
            raise ValueError('Empty source T1/T2 point cloud')
        supervision_t1, supervision_t2 = self._load_point_supervision_pair(record, p1, p2)
        bounds = self._choose_point_bounds(record, p1, p2, reference_bounds=reference_bounds)
        source_n1 = int(p1['coord'].shape[0])
        source_n2 = int(p2['coord'].shape[0])
        if bounds is not None:
            p1 = crop_points_xy(p1, bounds)
            p2 = crop_points_xy(p2, bounds)
            supervision_t1 = crop_point_supervision(supervision_t1, p1['_crop_mask'])
            supervision_t2 = crop_point_supervision(supervision_t2, p2['_crop_mask'])
        else:
            p1['_source_indices'] = torch.arange(source_n1, dtype=torch.long)
            p2['_source_indices'] = torch.arange(source_n2, dtype=torch.long)
        n1 = int(p1['coord'].shape[0])
        n2 = int(p2['coord'].shape[0])
        if n1 == 0 or n2 == 0:
            raise ValueError(f"{self.spec.name}/{record.get('id')}: empty T1/T2 point cloud after common spatial crop; bounds={bounds}, N1={n1}, N2={n2}")
        if supervision_t1['semantic'].shape[0] != n1 or supervision_t1['event'].shape[0] != n1:
            raise RuntimeError('T1 crop broke point/label topology')
        if supervision_t2['semantic'].shape[0] != n2 or supervision_t2['event'].shape[0] != n2:
            raise RuntimeError('T2 crop broke point/label topology')
        common_z0 = min(float(p1['coord'][:, 2].min().item()), float(p2['coord'][:, 2].min().item()))
        if bounds is not None:
            shared_xyz_origin = (float(bounds[0]), float(bounds[1]), common_z0)
        else:
            common_min = torch.minimum(p1['coord'].amin(dim=0), p2['coord'].amin(dim=0))
            shared_xyz_origin = tuple((float(v) for v in common_min.tolist()))
        target = build_bitemporal_point_target(supervision_t1, supervision_t2, class_names=self.spec.class_names, ignored_id=self.spec.ignored_id)
        return {'point_dict_t1': make_point_dict(p1, shared_xyz_origin=shared_xyz_origin), 'point_dict_t2': make_point_dict(p2, shared_xyz_origin=shared_xyz_origin), 'point_crs_t1': p1.get('crs'), 'point_crs_t2': p2.get('crs'), 'point_source_indices_t1': p1['_source_indices'], 'point_source_indices_t2': p2['_source_indices'], 'point_source_count_t1': source_n1, 'point_source_count_t2': source_n2, 'bounds': bounds, 'shared_xyz_origin': shared_xyz_origin, '_point_target': target}

    def _load_targets(self, record):
        if self.target_builder is not None:
            return self.target_builder(record, self.spec)
        s1 = read_label_array(self._path(record['semantic_t1'])) if 'semantic_t1' in record else None
        s2 = read_label_array(self._path(record['semantic_t2'])) if 'semantic_t2' in record else None
        ch = read_label_array(self._path(record['change'])) if 'change' in record else None
        target = build_canonical_target(label_mode=self.spec.label_mode, semantic_t1=s1, semantic_t2=s2, change=ch, class_names=self.spec.class_names, ignored_id=self.spec.ignored_id)
        out = {'change': target.change, 'semantic_t1': target.semantic_t1, 'semantic_t2': target.semantic_t2}
        if target.change_valid is not None:
            out['change_valid'] = target.change_valid
        if target.semantic_valid_t1 is not None:
            out['semantic_valid_t1'] = target.semantic_valid_t1
        if target.semantic_valid_t2 is not None:
            out['semantic_valid_t2'] = target.semantic_valid_t2
        return out

    def __getitem__(self, index):
        record = self.records[index]
        sample: Dict[str, Any] = {'sample_id': record.get('id', str(index)), 'dataset_name': self.spec.name, 'route': self.route, 'task_mode': self.route, 'prompt': self.prompt, 'class_names': dict(self.spec.class_names)}
        if self.spec.has_image:
            sample.update(self._load_2d(record))
        point_target = None
        if self.spec.has_point:
            reference_bounds = None
            if self.route == '2d3d':
                geo_t1 = sample.get('geo_t1')
                geo_t2 = sample.get('geo_t2')
                bounds_t1 = None if geo_t1 is None else geo_t1.get('bounds')
                bounds_t2 = None if geo_t2 is None else geo_t2.get('bounds')
                if bounds_t1 is not None and bounds_t2 is not None:
                    if not np.allclose(np.asarray(bounds_t1, dtype=np.float64), np.asarray(bounds_t2, dtype=np.float64), atol=1e-06, rtol=0.0):
                        raise ValueError('T1/T2 image world bounds differ in a 2D+3D sample')
                    reference_bounds = tuple((float(v) for v in bounds_t1))
                elif record.get('bounds') is not None:
                    reference_bounds = tuple((float(v) for v in record['bounds']))
            point_part = self._load_3d(record, reference_bounds=reference_bounds)
            point_target = point_part.pop('_point_target')
            sample.update(point_part)
        if self.spec.has_point:
            sample['target'] = point_target
        else:
            sample['target'] = self._load_targets(record)
        if self.spec.has_image and (not self.spec.has_point):
            h, w = sample['images_t1'].shape[-2:]
            for key in ('change', 'semantic_t1', 'semantic_t2'):
                target = sample['target'][key]
                if target.ndim >= 2 and tuple(target.shape[-2:]) != (h, w):
                    raise ValueError(f"{self.spec.name}/{sample['sample_id']}: {key} shape {tuple(target.shape)} does not match image {(h, w)}")
        if self.spec.has_point:
            n1 = sample['point_dict_t1']['coord'].shape[0]
            n2 = sample['point_dict_t2']['coord'].shape[0]
            for key in ('semantic_t1', 'event_t1'):
                if sample['target'][key].shape[0] != n1:
                    raise RuntimeError(f"{self.spec.name}/{sample['sample_id']}: {key} length {sample['target'][key].shape[0]} does not match T1 points {n1}")
            for key in ('semantic_t2', 'event_t2'):
                if sample['target'][key].shape[0] != n2:
                    raise RuntimeError(f"{self.spec.name}/{sample['sample_id']}: {key} length {sample['target'][key].shape[0]} does not match T2 points {n2}")
            for key in ('semantic_valid_t1', 'event_valid_t1'):
                if key in sample['target'] and sample['target'][key].shape[0] != n1:
                    raise RuntimeError(f"{self.spec.name}/{sample['sample_id']}: {key} length {sample['target'][key].shape[0]} does not match T1 points {n1}")
            for key in ('semantic_valid_t2', 'event_valid_t2'):
                if key in sample['target'] and sample['target'][key].shape[0] != n2:
                    raise RuntimeError(f"{self.spec.name}/{sample['sample_id']}: {key} length {sample['target'][key].shape[0]} does not match T2 points {n2}")
        if self.route == '2d3d':
            raster_crs = sample['geo_t1'].get('crs') if sample.get('geo_t1') is not None else None
            known_crs = [c for c in (raster_crs, sample.get('point_crs_t1'), sample.get('point_crs_t2')) if c]
            if known_crs and any((c != known_crs[0] for c in known_crs[1:])):
                raise ValueError('2D+3D sample contains mismatched CRS metadata. Prepare all modalities in one CRS before training.')
        sample['spatial_meta'] = {'bounds': sample.get('bounds', record.get('bounds')), 'raster_geo_t1': sample.get('geo_t1'), 'raster_geo_t2': sample.get('geo_t2'), 'point_crs_t1': sample.get('point_crs_t1'), 'point_crs_t2': sample.get('point_crs_t2'), 'point_window_size_m': PAIR_POINT_WINDOW_SIZE_M, 'shared_xyz_origin': sample.get('shared_xyz_origin'), 'point_source_count_t1': sample.get('point_source_count_t1'), 'point_source_count_t2': sample.get('point_source_count_t2')}
        return sample

def _self_test():
    classes = {0: 'ground', 1: 'building', 2: 'vegetation', 3: 'clutter'}
    s1 = torch.tensor([[0, 1], [2, 3]])
    s2 = torch.tensor([[0, 2], [2, 3]])
    out = build_canonical_target(label_mode='semantic_pair', semantic_t1=s1, semantic_t2=s2, class_names=classes)
    assert torch.equal(out.change, torch.tensor([[0, 1], [0, 0]]))
    assert out.change_valid is None
    out = build_canonical_target(label_mode='semantic_pair', semantic_t1=torch.tensor([0, 9, 1]), semantic_t2=torch.tensor([0, 1, 1]), class_names={0: 'ground', 1: 'building'}, ignored_id=9)
    assert out.semantic_valid_t1.tolist() == [True, False, True]
    assert out.change_valid.tolist() == [True, False, True]
    target = build_bitemporal_point_target({'semantic': torch.tensor([0, 1, 2]), 'event': torch.tensor([0, 2, 0]), 'event_valid': torch.tensor([True, True, False])}, {'semantic': torch.tensor([0, 3]), 'event': torch.tensor([1, 0]), 'event_valid': torch.tensor([True, True])}, class_names=classes)
    assert target['semantic_t1'].shape == (3,)
    assert target['semantic_t2'].shape == (2,)
    assert target['event_t1'].tolist() == [0, 2, 0]
    assert target['event_t2'].tolist() == [1, 0]
    assert target['event_valid_t1'].tolist() == [True, True, False]
    assert 'change_t1' not in target and 'change_t2' not in target
    point = {'coord': torch.tensor([[10.0, 20.0, 1.0], [11.0, 21.0, 2.0]]), 'rgb': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 'intensity': torch.tensor([[7.0], [8.0]])}
    packed = make_point_dict(point, shared_xyz_origin=(10.0, 20.0, 1.0))
    assert set(packed) == {'coord', 'rgb', 'intensity'}
    assert torch.equal(packed['coord'], torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    geometry_only = make_point_dict({'coord': point['coord']}, shared_xyz_origin=(10.0, 20.0, 1.0))
    assert set(geometry_only) == {'coord'}
    assert DatasetSpec.__dataclass_fields__['ignored_id'].default is None
    assert PAIR_EVENT_NUM_CLASSES == 6
    assert PAIR_POINT_WINDOW_SIZE_M == 51.2
    print('pair_dataset.py self-test: PASS')
if __name__ == '__main__':
    _self_test()
