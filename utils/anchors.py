import numpy as np

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):  # 这里的宽、高参数为特征图的宽高
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # print(type(shift_x)) <class 'numpy.ndarray'>
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel(),), axis=1)
    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]  # A=9
    K = shift.shape[0]  # K=1444
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()
    print(nine_anchors)
    height, width, feat_stride = 38, 38, 16
    anchors_all = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x, shift_y)
    box_widths = anchors_all[:, 2]-anchors_all[:, 0]
    box_heights = anchors_all[:, 3]-anchors_all[:, 1]
    
    for i in [9000,9001,9002,9003,9004,9005,9006,9007,9008]:
        rect = plt.Rectangle([anchors_all[i, 0], anchors_all[i, 1]], box_widths[i], box_heights[i], color="r", fill=False)
        ax.add_patch(rect)
    plt.show()