MARGIN = 25
DEPTH, HEIGHT, WIDTH = 112, 280, 280

UPPER = -1
LOWER = 1

NUM_CLASSES = 1

RESIZE_DEPTH = 64
RESIZE_HEIGHT = 128
RESIZE_WIDTH = 128

TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

UPPER_TOOTH_NUM = [
    '21', '22', '23', '24', '25', '26', '27', '28',
    '11', '12', '13', '14', '15', '16', '17', '18', 
]

LOWER_TOOTH_NUM = [
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]

# Legacy training/evaluation lists.
# The public release focuses on inference, so these lists are intentionally empty.
# Populate them with case identifiers if you reuse the legacy dataloader.
METAL_TRAIN = []
METAL_TEST = []
