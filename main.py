from PIL import Image
import numpy as np

from src.vision import FloorProjectiveTransform

image = np.array(Image.open("./sources/imgs/red_c.png").convert("RGB"))

T = FloorProjectiveTransform.from_points()

result = T.estimate_boundaries(
    image,
    min_component_area=50,
    bottom_band_px=1,
    max_gap_below_px=8,
    dbscan_eps=0.05,
    dbscan_min_samples=4,
    min_cluster_points=10,
)

robot_points = result.robot_points
image_pixels = result.image_pixels

T.plot_result(image, result)

# For conversion to global world coordinates
# world_points, result = T.estimate_boundaries_world(
#     image,
#     robot_pose=R,
#     max_gap_below_px=8,
# )