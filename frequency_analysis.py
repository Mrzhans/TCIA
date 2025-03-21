import numpy as np
import cv2  # 用于加载图像
import matplotlib.pyplot as plt

def plot_frequency_spectrum(image, title):
    # 1. 对灰度图应用 2D 傅里叶变换并将频谱中心移到图像中央
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)

    # 2. 计算幅度谱，并取对数以便可视化（避免数值范围过大）
    magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-8)

    # 3. 绘制频谱图
    plt.figure(figsize=(6, 5))
    plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(f'Frequency Spectrum of {title}')
    plt.axis('off')  # 不显示轴
    plt.show()

# 加载灰度图像：假设已生成的图像路径如下
images = {
    'Original': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\00390.png',
    'LL': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\vis_y_ll.png',
    'LH': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\vis_y_lh_10.png',
    'HL': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\vis_y_hl_10.png',
    'HH': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\vis_y_hh_10.png'
}

# 对每张图片进行频谱分析
for title, path in images.items():
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
    if image is None:
        print(f"Failed to load {title} from {path}")
        continue
    plot_frequency_spectrum(image, title)


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# def plot_frequency_curve(image, title):
#     # 1. 对灰度图应用 2D 傅里叶变换并将频谱中心移到中央
#     f_transform = np.fft.fft2(image)
#     f_shifted = np.fft.fftshift(f_transform)
    
#     # 2. 计算幅度谱，并取对数平滑数值
#     magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-8)

#     # 3. 计算沿X轴和Y轴的平均频谱曲线
#     x_mean = np.mean(magnitude_spectrum, axis=0)  # 沿Y轴取平均（水平频率）
#     y_mean = np.mean(magnitude_spectrum, axis=1)  # 沿X轴取平均（垂直频率）

#     # 4. 绘制频谱曲线
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].plot(x_mean)
#     axes[0].set_title(f'Frequency Curve (X-axis) of {title}')
#     axes[0].set_xlabel('Frequency')
#     axes[0].set_ylabel('Magnitude')

#     axes[1].plot(y_mean)
#     axes[1].set_title(f'Frequency Curve (Y-axis) of {title}')
#     axes[1].set_xlabel('Frequency')
#     axes[1].set_ylabel('Magnitude')

#     plt.show()

# # 加载灰度图像
# images = {
#     'Original': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\00390.png',
#     'LL': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\ir_ll.png',
#     'LH': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\ir_lh_10.png',
#     'HL': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\ir_hl_10.png',
#     'HH': r'C:\Users\13743\Desktop\QWBNet\Wavlet Pool Transform\ir_hh_10.png'
# }

# # 对每张图像进行频率曲线分析
# for title, path in images.items():
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         print(f"Failed to load {title} from {path}")
#         continue
#     plot_frequency_curve(image, title)



# import torch
# import cv2
# import numpy as np
# import os
# from sklearn.manifold import TSNE
# from utils.img_read_save import image_read_cv2
# from quaternion.qfnet_v4 import get_wav
# import matplotlib.pyplot as plt


# def load_wavelet_components(vis_y_path, ir_path):
#     vis_y = torch.FloatTensor(cv2.split(image_read_cv2(vis_y_path, mode='YCrCb'))[0]).unsqueeze(0).to('cuda')
#     ir = torch.FloatTensor(image_read_cv2(ir_path, mode='GRAY')).unsqueeze(0).to('cuda')
    
#     LL, LH, HL, HH = get_wav(in_channels=1)
#     # vis_components = [LL(vis_y), LH(vis_y), HL(vis_y), HH(vis_y)]
#     # ir_components = [LL(ir), LH(ir), HL(ir), HH(ir)]
#     vis_components = [LL(vis_y)]
#     ir_components = [LL(ir)]
    
#     # vis_components = [LH(vis_y), HL(vis_y), HH(vis_y)]
#     # ir_components = [LH(ir), HL(ir), HH(ir)]
    
#     return vis_components, ir_components


# def prepare_features(components):
#     feature_vectors = []
#     for comp in components:
#         comp_np = comp.squeeze(0).cpu().detach().numpy()
#         # 将特征向量展平
#         comp_flatten = comp_np.flatten()
#         feature_vectors.append(comp_flatten)
        
#     # 确保所有特征向量具有相同的形状
#     max_length = max(len(vec) for vec in feature_vectors)  # 找到最长的特征向量
#     feature_vectors_fixed = np.array([np.pad(vec, (0, max_length - len(vec)), 'constant') for vec in feature_vectors])  # 用零填充不足的特征向量
    
#     return feature_vectors_fixed



# def visualize_tsne(vis_features, ir_features):
#     features = np.vstack([vis_features, ir_features]).astype(np.float32)
#     tsne = TSNE(n_components=2, random_state=0, perplexity=30)  # 调整perplexity值
#     transformed = tsne.fit_transform(features)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(transformed[:len(vis_features), 0], transformed[:len(vis_features), 1], c='blue', label='Visible', marker='o')
#     plt.scatter(transformed[len(vis_features):, 0], transformed[len(vis_features):, 1], c='red', label='Infrared', marker='x')

#     plt.axis('off')
#     plt.legend()
    
#     plt.savefig('tsne_visualization_LL.jpg', format='jpg', dpi=300)  # 保存图像
#     plt.show()  # 显示图像


# # 读取图像并处理
# vis_path = r'C:\Users\13743\Desktop\Datasets\M3FD\Vis'
# ir_path = r'C:\Users\13743\Desktop\Datasets\M3FD\Ir'

# vis_components_list = []
# ir_components_list = []

# # 读取所有图像
# for image_name in os.listdir(vis_path):
#     if image_name.endswith('.png'):  # 确保只处理png文件
#         vis_components, ir_components = load_wavelet_components(
#             os.path.join(vis_path, image_name),
#             os.path.join(ir_path, image_name)
#         )
#         vis_components_list.extend(vis_components)
#         ir_components_list.extend(ir_components)

# # 准备特征向量
# vis_features = prepare_features(vis_components_list)
# ir_features = prepare_features(ir_components_list)

# # t-SNE 可视化
# visualize_tsne(vis_features, ir_features)

