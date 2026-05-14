from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from skimage.transform import ProjectiveTransform
from sklearn.cluster import DBSCAN


@dataclass
class BoundaryResult:
    """Result of object-floor boundary extraction."""

    robot_points: np.ndarray          # shape (N, 2), points in the calibrated floor frame
    image_pixels: np.ndarray          # shape (N, 2), image pixels as (x, y)
    red_mask: np.ndarray              # uint8 mask, 0/255
    floor_mask: np.ndarray            # uint8 mask, 0/255
    contact_mask: np.ndarray          # uint8 mask, 0/255
    image_component_labels: np.ndarray # label per returned image pixel
    world_cluster_labels: np.ndarray   # DBSCAN label per returned point, -1 means removed/noise


class FloorProjectiveTransform(ProjectiveTransform):
    """
    Homography from image pixels to the floor plane.

    The transform is still a normal skimage.transform.ProjectiveTransform:
        robot_xy = T(image_pixels)

    Extra methods are added for calibration, object-floor boundary extraction,
    cluster filtering, and plotting.
    """

    DEFAULT_IMAGE_POINTS = np.array(
        [
            [326, 243],
            [196, 242],
            [460, 244],
            [326, 214],
            [414, 215],
            [239, 214],
            [153, 214],
            [71, 213],
            [595, 216],
        ],
        dtype=np.float64,
    )

    DEFAULT_FLOOR_POINTS = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, -0.5],
            [1.5, 0.0],
            [1.5, -0.5],
            [1.5, 0.5],
            [1.5, 1.0],
            [1.5, 1.5],
            [1.5, -1.5],
        ],
        dtype=np.float64,
    )

    @classmethod
    def from_points(
        cls,
        image_points: np.ndarray | None = None,
        floor_points: np.ndarray | None = None,
    ) -> "FloorProjectiveTransform":
        """Estimate image -> floor homography from corresponding points."""
        if image_points is None and floor_points is None:
            image_points = cls.DEFAULT_IMAGE_POINTS
            floor_points = cls.DEFAULT_FLOOR_POINTS
        elif image_points is None or floor_points is None:
            raise ValueError("Both image_points and floor_points must be provided.")

        image_points = np.asarray(image_points, dtype=np.float64)
        floor_points = np.asarray(floor_points, dtype=np.float64)

        if image_points.shape != floor_points.shape or image_points.shape[1] != 2:
            raise ValueError(
                "image_points and floor_points must both have shape (N, 2)."
            )
        if len(image_points) < 4:
            raise ValueError("At least 4 point correspondences are required.")

        # scikit-image >= 0.26 prefers from_estimate(); older versions use estimate().
        if hasattr(cls, "from_estimate"):
            transform = cls.from_estimate(image_points, floor_points)
        else:
            transform = cls()
            ok = transform.estimate(image_points, floor_points)
            if not ok:
                raise ValueError("Failed to estimate image -> floor homography.")

        transform.image_points = image_points
        transform.floor_points = floor_points
        return transform

    def image_to_floor(self, pixels: np.ndarray) -> np.ndarray:
        """Map image pixels, given as (x, y), to floor coordinates."""
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return self(pixels)

    def floor_to_image(self, floor_points: np.ndarray) -> np.ndarray:
        """Map floor coordinates back to image pixels."""
        floor_points = np.asarray(floor_points, dtype=np.float64)
        if floor_points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return self.inverse(floor_points)

    def reprojection_error(self) -> np.ndarray:
        """Return calibration error in floor coordinates for stored calibration points."""
        if not hasattr(self, "image_points") or not hasattr(self, "floor_points"):
            raise ValueError("No stored calibration points. Use FloorProjectiveTransform.from_points().")
        pred = self.image_to_floor(self.image_points)
        return np.linalg.norm(pred - self.floor_points, axis=1)

    @staticmethod
    def red_mask(
        image: np.ndarray,
        lower_red1: tuple[int, int, int] = (0, 80, 80),
        upper_red1: tuple[int, int, int] = (10, 255, 255),
        lower_red2: tuple[int, int, int] = (170, 80, 80),
        upper_red2: tuple[int, int, int] = (179, 255, 255),
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Segment red objects in an RGB image using two HSV hue intervals."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        mask1 = cv2.inRange(
            hsv,
            np.array(lower_red1, dtype=np.uint8),
            np.array(upper_red1, dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array(lower_red2, dtype=np.uint8),
            np.array(upper_red2, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask1, mask2)

        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    @staticmethod
    def floor_mask(
        image: np.ndarray,
        object_mask: np.ndarray,
        max_floor_saturation: int = 70,
        min_floor_value: int = 70,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Approximate visible floor mask for the current synthetic scene.

        The floor is mostly low-saturation and sufficiently bright. Red object pixels
        are explicitly removed. For a real robot, this method should be replaced by
        a more robust semantic/depth-based floor segmentation.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        floor = (
            (hsv[..., 1] <= max_floor_saturation)
            & (hsv[..., 2] >= min_floor_value)
            & (object_mask == 0)
        )
        floor = floor.astype(np.uint8) * 255

        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, kernel)
            floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, kernel)

        return floor

    @staticmethod
    def _has_support_below(
        support_mask: np.ndarray,
        x: int,
        y: int,
        max_gap_px: int,
        lateral_px: int,
    ) -> bool:
        h, w = support_mask.shape[:2]
        y0 = min(y + 1, h)
        y1 = min(y + max_gap_px + 1, h)
        x0 = max(x - lateral_px, 0)
        x1 = min(x + lateral_px + 1, w)

        if y0 >= y1:
            return False
        return bool(np.any(support_mask[y0:y1, x0:x1] > 0))

    @staticmethod
    def component_bottom_contact_pixels(
        object_mask: np.ndarray,
        support_mask: np.ndarray,
        min_component_area: int = 50,
        bottom_band_px: int = 1,
        max_gap_below_px: int = 4,
        lateral_support_px: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract bottom contact pixels from each red connected component separately.

        Important detail: the bottom-pixel selection is component-wise, not global.
        This keeps several visible object boundaries even when different objects or
        wall segments overlap in the image x-direction.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(object_mask)

        pixels: list[np.ndarray] = []
        component_ids: list[int] = []
        contact_mask = np.zeros_like(object_mask, dtype=np.uint8)

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue

            ys, xs = np.where(labels == label_id)
            if len(xs) == 0:
                continue

            for x in np.unique(xs):
                col_ys = ys[xs == x]
                y_bottom = int(col_ys.max())

                if not FloorProjectiveTransform._has_support_below(
                    support_mask=support_mask,
                    x=int(x),
                    y=y_bottom,
                    max_gap_px=max_gap_below_px,
                    lateral_px=lateral_support_px,
                ):
                    continue

                y_min = max(y_bottom - bottom_band_px + 1, int(col_ys.min()))
                band_ys = col_ys[col_ys >= y_min]
                band_pixels = np.column_stack(
                    [np.full_like(band_ys, x), band_ys]
                ).astype(np.int32)

                pixels.append(band_pixels)
                component_ids.extend([label_id] * len(band_pixels))
                contact_mask[band_ys, x] = 255

        if not pixels:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                contact_mask,
            )

        return (
            np.vstack(pixels).astype(np.int32),
            np.asarray(component_ids, dtype=np.int32),
            contact_mask,
        )

    @staticmethod
    def filter_world_clusters(
        points: np.ndarray,
        eps: float = 0.05,
        min_samples: int = 4,
        min_cluster_points: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove isolated projected points in floor coordinates using DBSCAN.

        Returns filtered points and a label vector aligned with the original points.
        Points removed as noise have label -1.
        """
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            return points.reshape(0, 2), np.empty((0,), dtype=np.int32)

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            if np.sum(labels == cluster_id) < min_cluster_points:
                labels[labels == cluster_id] = -1

        return points[labels != -1], labels.astype(np.int32)

    def estimate_boundaries(
        self,
        image: np.ndarray,
        *,
        min_component_area: int = 50,
        bottom_band_px: int = 1,
        max_gap_below_px: int = 4,
        lateral_support_px: int = 1,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 4,
        min_cluster_points: int = 10,
        require_floor_below: bool = True,
    ) -> BoundaryResult:
        """
        Detect red object-floor boundaries and project them to the calibrated floor frame.

        If require_floor_below=True, a red pixel is accepted as contact only when a
        visible floor pixel exists just below it. This strongly reduces false edges
        from internal vertical gradients and object silhouettes not lying on the floor.
        """
        object_mask = self.red_mask(image)

        if require_floor_below:
            support_mask = self.floor_mask(image, object_mask)
        else:
            support_mask = cv2.bitwise_not(object_mask)

        image_pixels, image_component_labels, contact_mask = self.component_bottom_contact_pixels(
            object_mask=object_mask,
            support_mask=support_mask,
            min_component_area=min_component_area,
            bottom_band_px=bottom_band_px,
            max_gap_below_px=max_gap_below_px,
            lateral_support_px=lateral_support_px,
        )

        robot_points_raw = self.image_to_floor(image_pixels)
        robot_points, world_cluster_labels = self.filter_world_clusters(
            robot_points_raw,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            min_cluster_points=min_cluster_points,
        )

        keep = world_cluster_labels != -1
        return BoundaryResult(
            robot_points=robot_points,
            image_pixels=image_pixels[keep],
            red_mask=object_mask,
            floor_mask=support_mask,
            contact_mask=contact_mask,
            image_component_labels=image_component_labels[keep],
            world_cluster_labels=world_cluster_labels[keep],
        )

    @staticmethod
    def robot_to_world(robot_points: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """
        Transform robot-frame floor points to global world coordinates.

        robot_pose can be a 3x3 homogeneous transform. Returned array has shape (N, 2).
        """
        robot_points = np.asarray(robot_points, dtype=np.float64)
        robot_pose = np.asarray(robot_pose, dtype=np.float64)

        if robot_points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if robot_pose.shape != (3, 3):
            raise ValueError("robot_pose must be a 3x3 homogeneous transform matrix.")

        robot_h = np.column_stack([robot_points, np.ones(len(robot_points))])
        world_h = robot_h @ robot_pose.T

        w = world_h[:, 2:3]
        safe_w = np.where(np.abs(w) < 1e-12, 1.0, w)
        return world_h[:, :2] / safe_w

    def estimate_boundaries_world(
        self,
        image: np.ndarray,
        robot_pose: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, BoundaryResult]:
        """Estimate boundary points and convert them from robot frame to global world frame."""
        result = self.estimate_boundaries(image, **kwargs)
        return self.robot_to_world(result.robot_points, robot_pose), result

    @staticmethod
    def plot_result(
        image: np.ndarray,
        result: BoundaryResult,
        title: str = "Object floor boundary",
    ) -> None:
        """Quick diagnostic plot: image overlay, masks, and floor-frame map."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].imshow(image)
        if len(result.image_pixels) > 0:
            axes[0].scatter(
                result.image_pixels[:, 0],
                result.image_pixels[:, 1],
                s=4,
            )
        axes[0].set_title("Detected contact pixels")
        axes[0].axis("off")

        axes[1].imshow(result.red_mask, cmap="gray")
        axes[1].set_title("Red object mask")
        axes[1].axis("off")

        axes[2].imshow(result.floor_mask, cmap="gray")
        axes[2].set_title("Support/floor mask")
        axes[2].axis("off")

        axes[3].set_title(title)
        if len(result.robot_points) > 0:
            axes[3].scatter(result.robot_points[:, 0], result.robot_points[:, 1], s=5)
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")
        axes[3].grid(True)
        axes[3].axis("equal")

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Backward-compatible functional API
# -----------------------------------------------------------------------------


def get_transform(img_p: np.ndarray | None = None, world_p: np.ndarray | None = None) -> FloorProjectiveTransform:
    """Backward-compatible wrapper. Returns image -> floor transform."""
    return FloorProjectiveTransform.from_points(img_p, world_p)


def get_object_floor_boundary_world(
    image: np.ndarray,
    T: FloorProjectiveTransform,
    *,
    min_component_area: int = 50,
    bottom_band_px: int = 1,
    max_gap_below_px: int = 4,
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 4,
    min_cluster_points: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper.

    Returns robot/floor-frame points, image pixels, and red mask. The name is kept
    only to avoid breaking existing notebooks.
    """
    result = T.estimate_boundaries(
        image,
        min_component_area=min_component_area,
        bottom_band_px=bottom_band_px,
        max_gap_below_px=max_gap_below_px,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        min_cluster_points=min_cluster_points,
    )
    return result.robot_points, result.image_pixels, result.red_mask


def find_connected_components(
    world_points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 4,
    min_cluster_points: int = 10,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Backward-compatible DBSCAN wrapper."""
    filtered_points, labels = FloorProjectiveTransform.filter_world_clusters(
        world_points,
        eps=eps,
        min_samples=min_samples,
        min_cluster_points=min_cluster_points,
    )

    connected_regions: dict[int, np.ndarray] = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        connected_regions[int(cluster_id)] = np.asarray(world_points)[labels == cluster_id]

    return filtered_points, connected_regions


def image2robot(image: np.ndarray, T: FloorProjectiveTransform) -> np.ndarray:
    """Return object boundary points in the robot/floor frame."""
    return T.estimate_boundaries(image).robot_points


def image2world(image: np.ndarray, T: FloorProjectiveTransform, R: np.ndarray) -> np.ndarray:
    """Return object boundary points in the global world frame."""
    robot_coords = image2robot(image, T)
    return FloorProjectiveTransform.robot_to_world(robot_coords, R)
