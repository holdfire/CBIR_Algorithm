import cv2
import numpy as np


def Canny(img):

    # sobel filter
    def sobel_filter(img, K_size=3):
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            H, W = img.shape

        # Zero padding
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        ## Sobel vertical
        Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        ## Sobel horizontal
        Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

        # filtering
        for y in range(H):
            for x in range(W):
                out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
                out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

        a = np.min(out_v)
        b = np.max(out_v)
        # 修改1：把梯度强度缩放到[-127.5, 127.5]之间
        out_v = (out_v - np.min(out_v)) * 255 / (np.max(out_v) - np.min(out_v)) - 127.5
        out_h = (out_h - np.min(out_h)) * 255 / (np.max(out_h) - np.min(out_h)) - 127.5
        out_v = out_v[pad: pad + H, pad: pad + W].astype(np.int8)
        out_h = out_h[pad: pad + H, pad: pad + W].astype(np.int8)

        return out_v, out_h

    def get_edge_angle(fx, fy):
        # get edge strength
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        # 修改2：把边缘edge线性缩放到[0, 255]
        edge = (1.414 * edge)
        edge = edge.astype(np.uint8)

        fx = np.maximum(fx, 1e-10)
        # fx[np.abs(fx) <= 1e-5] = 1e-5
        # get edge angle，此时的angle在[-1.57, 1.57]之间
        angle = np.arctan(fy / fx)

        # cv2.imwrite("edge.png", edge)
        # cv2.imwrite("angle.png", angle)
        # cv2.imshow("edge.png", edge)
        # cv2.imshow("angle.png", angle)
        # cv2.waitKey(0)
        return edge, angle

    def angle_quantization(angle):
        angle = angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135
        return _angle

    def non_maximum_suppression(angle, edge):
        H, W = angle.shape
        _edge = edge.copy()

        for y in range(H):
            for x in range(W):
                if angle[y, x] == 0:
                    dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                elif angle[y, x] == 45:
                    dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                elif angle[y, x] == 90:
                    dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                elif angle[y, x] == 135:
                    dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                if x == 0:
                    dx1 = max(dx1, 0)
                    dx2 = max(dx2, 0)
                if x == W - 1:
                    dx1 = min(dx1, 0)
                    dx2 = min(dx2, 0)
                if y == 0:
                    dy1 = max(dy1, 0)
                    dy2 = max(dy2, 0)
                if y == H - 1:
                    dy1 = min(dy1, 0)
                    dy2 = min(dy2, 0)
                if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                    _edge[y, x] = 0
        # cv2.imwrite("edge.png", _edge)
        return _edge

    def hysterisis(edge, HT=100, LT=30):
        H, W = edge.shape

        # Histeresis threshold
        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
        _edge[1: H + 1, 1: W + 1] = edge

        ## 8 - Nearest neighbor
        nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

        for y in range(1, H + 2):
            for x in range(1, W + 2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                if np.max(_edge[y - 1:y + 2, x - 1:x + 2] * nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0

        edge = _edge[1:H + 1, 1:W + 1]
        # cv2.imwrite('edge.png', edge)
        return edge

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gaussian filtering
    gray = cv2.medianBlur(gray, 3)
    # sobel filtering
    fy, fx = sobel_filter(gray, K_size=3)
    # get edge strength, angle
    edge, angle = get_edge_angle(fx, fy)
    # angle quantization
    angle = angle_quantization(angle)
    # non maximum suppression
    edge = non_maximum_suppression(angle, edge)
    # hysterisis threshold
    edge = hysterisis(edge, 50, 20)

    return edge, angle


def Hough_Line(edge, img):

    ## Voting
    def voting(edge):
        H, W = edge.shape

        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

        # hough table
        hough = np.zeros((rho_max * 2, 180), dtype=np.int)

        # get index of edge
        ind = np.where(edge == 255)

        ## hough transformation
        for y, x in zip(ind[0], ind[1]):
            for theta in range(0, 180, dtheta):
                # get polar coordinat4s
                t = np.pi / 180 * theta
                rho = int(x * np.cos(t) + y * np.sin(t))

                # vote
                hough[rho + rho_max, theta] += 1

        out = hough.astype(np.uint8)

        return out

    # non maximum suppression
    def non_maximum_suppression(hough):
        rho_max, _ = hough.shape

        ## non maximum suppression
        for y in range(rho_max):
            for x in range(180):
                # get 8 nearest neighbor
                x1 = max(x - 1, 0)
                x2 = min(x + 2, 180)
                y1 = max(y - 1, 0)
                y2 = min(y + 2, rho_max - 1)
                if np.max(hough[y1:y2, x1:x2]) == hough[y, x] and hough[y, x] != 0:
                    pass
                # hough[y,x] = 255
                else:
                    hough[y, x] = 0

        return hough

    def inverse_hough(hough, img):
        H, W, _ = img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180 - rho_max / 2

        # each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta

            # hough -> (x,y)
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out[y, x] = [0, 0, 255]
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [0, 0, 255]

        out = out.astype(np.uint8)

        return out

    # voting
    hough = voting(edge)
    # non maximum suppression
    hough = non_maximum_suppression(hough)
    # inverse hough
    out = inverse_hough(hough, img)
    return out



# Read image
img = cv2.imread("../data/online/test/001.png").astype(np.float32)

# Canny
edge, angle = Canny(img)
edge = edge.astype(np.uint8)
angle = angle.astype(np.uint8)
# # Save result
# cv2.imwrite("edge.jpg", edge)
# cv2.imshow("edge", edge)
# cv2.imwrite("angle.jpg", angle)
# cv2.imshow("angle", angle)

# Hough line
line = Hough_Line(edge, img)
line = line.astype(np.uint8)
cv2.imwrite("mine.jpg", line)

cv2.waitKey(0)
cv2.destroyAllWindows()