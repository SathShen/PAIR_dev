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

    class_names={
        0: "unchanged",
        1: "water",
        2: "ground",
        3: "low vegetation",
        4: "tree",
        5: "building",
        6: "sports field",
    },

    image_size=512,
    ignore_value=None,
)

LANDSATSCD_SPEC = DatasetSpec(
    name="LandsatSCD",
    task_mode="2d",
    label_mode="semantic_pair",
    class_names={
        0: "unchanged",
        1: "farmland",
        2: "desert",
        3: "building",
        4: "water",
    },

    image_size=512,
    ignore_value=None,
)
