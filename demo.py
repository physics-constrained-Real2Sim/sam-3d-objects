import sys

# import inference code
sys.path.append("notebook")
from notebook.inference import Inference, load_image, load_single_mask
import numpy as np
# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
# inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

print("image shape:", image.shape)
print("mask shape:", mask.shape)

print(np.unique(image))
print(np.unique(mask))
# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")

