import numpy as np

def depth_to_points_ZED(depth: np.ndarray, mask):

    points = np.zeros((depth.shape[0] * depth.shape[1], 3), dtype=np.float32)

    cx = 966.2852172851562
    cy = 531.3258666992188
    fx = 1387.484130859375
    fy = 1387.484130859375

    for idx, point in enumerate(points):
        # 从点数编号到点坐标索引
        pointX = (idx % depth.shape[1])
        pointY = int(idx / depth.shape[1])

        if mask[pointY, pointX] > 0:
            # set Point Plane
            z = depth[pointY, pointX]
            if z > 0:
                points[idx][0] = ((pointX - cx) * z) / (fx)
                points[idx][1] = ((pointY - cy) * z) / (fy)
                points[idx][2] = z
                # colors[idx][0] = rgb[pointY, pointX][2]
                # colors[idx][1] = rgb[pointY, pointX][1]
                # colors[idx][2] = rgb[pointY, pointX][0]

    return points