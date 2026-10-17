"""Planar homography calibration mapping image pixels to real-world field/diamond coordinates."""

from __future__ import annotations

from foundryvision.core.schema import FieldCoordinate, Point2D


Matrix3x3 = list[list[float]]


class PlanarFieldCalibrator:
    """Computes a 3x3 planar homography matrix H from >=4 field landmark correspondences."""

    def __init__(
        self,
        reference_landmarks: dict[str, tuple[float, float]],
        unit: str = "yards",
    ) -> None:
        self.reference_landmarks = reference_landmarks
        self.unit = unit
        self._homography: Matrix3x3 | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._homography is not None

    def update_from_keypoints(self, observed_keypoints: dict[str, Point2D]) -> bool:
        """Estimate homography H using Direct Linear Transform (DLT) on matched landmarks."""
        src_pts: list[tuple[float, float]] = []
        dst_pts: list[tuple[float, float]] = []
        for name, pt in observed_keypoints.items():
            if name in self.reference_landmarks and pt.confidence >= 0.3:
                src_pts.append((pt.x, pt.y))
                dst_pts.append(self.reference_landmarks[name])

        if len(src_pts) < 4:
            return self._homography is not None

        self._homography = self._compute_dlt_homography(src_pts, dst_pts)
        return self._homography is not None

    def project_point(self, pixel_xy: tuple[float, float]) -> FieldCoordinate | None:
        """Project a 2D pixel coordinate (u, v) to a 2D real-world FieldCoordinate (X, Y)."""
        if self._homography is None:
            return None
        px, py = pixel_xy
        h = self._homography
        w = h[2][0] * px + h[2][1] * py + h[2][2]
        if abs(w) < 1e-8:
            return None
        x = (h[0][0] * px + h[0][1] * py + h[0][2]) / w
        y = (h[1][0] * px + h[1][1] * py + h[1][2]) / w
        return FieldCoordinate(x=round(x, 3), y=round(y, 3), unit=self.unit)

    @staticmethod
    def _compute_dlt_homography(
        src: list[tuple[float, float]], dst: list[tuple[float, float]]
    ) -> Matrix3x3 | None:
        """Solve normal equations (A^T A) h = A^T b for planar homography with h33 = 1."""
        ata = [[0.0] * 8 for _ in range(8)]
        atb = [0.0] * 8
        for (x, y), (u, v) in zip(src, dst):
            row1 = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]
            row2 = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]
            for r, rhs in ((row1, u), (row2, v)):
                for i in range(8):
                    atb[i] += r[i] * rhs
                    for j in range(8):
                        ata[i][j] += r[i] * r[j]

        sol = PlanarFieldCalibrator._solve_linear_8x8(ata, atb)
        if sol is None:
            return None
        return [
            [sol[0], sol[1], sol[2]],
            [sol[3], sol[4], sol[5]],
            [sol[6], sol[7], 1.0],
        ]

    @staticmethod
    def _solve_linear_8x8(a: list[list[float]], b: list[float]) -> list[float] | None:
        n = 8
        aug = [row[:] + [b[idx]] for idx, row in enumerate(a)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) < 1e-11:
                return None
            aug[col], aug[pivot] = aug[pivot], aug[col]
            div = aug[col][col]
            for j in range(col, n + 1):
                aug[col][j] /= div
            for r in range(n):
                if r != col and abs(aug[r][col]) > 0.0:
                    factor = aug[r][col]
                    for j in range(col, n + 1):
                        aug[r][j] -= factor * aug[col][j]
        return [aug[i][n] for i in range(n)]
