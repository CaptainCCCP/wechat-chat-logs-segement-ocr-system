from PIL import Image
import matplotlib.pyplot as plt
Image.open('测试\PennFudanPed\PNGImages/FudanPed00001.png')

# 读取对应图片的掩码图片
mask = Image.open('测试\PennFudanPed\PedMasks\FudanPed00001_mask.png')

# 读取的mask默认为“L”模式，需要转换为“P”模式调用调色板函数
mask = mask.convert("P")

# 针对“P”类图片调用调色板函数
# 看来掩码图片存的不是RGB数值，而是类别index
mask.putpalette([
    0, 0, 0, # 0号像素为黑色
    255, 0, 0, # 1号像素为红色
    255, 255, 0, # 2号像素为黄色
    255, 153, 0, # 3号像素为黄色
])
plt.imshow(mask)
plt.axis('off')  # 关闭坐标轴
plt.show()