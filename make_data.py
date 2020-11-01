# 把图像二值化（降维）
# (60, 160)图片像素，灰度化后的,直接用灰度图（二维），黑白图是三维的


def black_white_ver():
    import cv2
    from verification_code_recognition import verification_code
    text, image = verification_code.gen_captcha_text_and_image()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('verification_code_img&label/picture_01/{0}.jpg'.format(text), gray_img) # 太多存不下
    return text, gray_img


# 制作数据集和标签
def con_array(*args):
    import numpy as np
    array_list = args[0]
    array_tol = np.array(array_list)
    print(array_tol.shape)
    return array_tol


def con_label(*args):
    import numpy as np
    label_array = np.array([])
    label_array = np.append(label_array, args)
    return label_array


# 存储
import numpy as np
# np.save()
# np.load()

# 50w训练数据，5w测试数据
img_list = []
label_list = []
run_list = [6250 for x in range(8)]
run1_list = [625 for x in range(8)]


def data_generator(a):
    for i in range(a):
        text, img = black_white_ver()
        img_list.append(img)
        label_list.append(text)


# 训练数据
data_generator(50000)
img_array = con_array(img_list)
label_array = con_label(label_list)
np.save('verification_code_img&label/train', img_array)
np.save('verification_code_img&label/train_label', label_array)
# 测试数据
img_list = []
label_list = []
data_generator(5000)
img_array = con_array(img_list)
label_array = con_label(label_list)
np.save('verification_code_img&label/test', img_array)
np.save('verification_code_img&label/test_label', label_array)



