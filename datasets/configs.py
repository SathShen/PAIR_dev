from datasets.pair_dataset import DatasetSpec

# ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
# ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
# LandsatSCD
ST_COLORMAP = [[255,255,255], [0,155,0], [255,165,0], [230,30,100], [0,170,240]]
ST_CLASSES = ['unchanged', 'farmland', 'desert', 'building', 'water']

SECOND_SPEC = DatasetSpec(
    name="SECOND",
    task_mode="2d",
    label_mode="semantic_pair",

    class_names=[
        "unchanged",   # 0
        "water",   # 1
        "ground",   # 2
        "low vegetation",   # 3
        "tree",   # 4
        "building",   # 5
        "sports field"   # 6
    ],

    image_size=512,
    ignore_value=None,
)

LANDSATSCD_SPEC = DatasetSpec(
    name="LandsatSCD",
    task_mode="2d",
    label_mode="semantic_pair",
    class_names=[
        "unchanged",   # 0
        "farmland",   # 1
        "desert",   # 2
        "building",   # 3
        "water"   # 4
    ],

    image_size=512,
    ignore_value=None,
)
