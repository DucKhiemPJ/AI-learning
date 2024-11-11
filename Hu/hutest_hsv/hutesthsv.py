import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy.stats import zscore

def extract_feature(image):
    # Trích xuất đặc trưng Hu Moments từ ảnh
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments.flatten()

def extract_hu_features_from_folder(path, label, output_binary_folder='binary_images'):
    list_of_files = os.listdir(path)
    features = []
    labels = []
    
    # Tạo thư mục lưu ảnh nhị phân nếu chưa tồn tại
    os.makedirs(output_binary_folder, exist_ok=True)

    for i in list_of_files:
        img = plt.imread(os.path.join(path, i))  # Đọc ảnh từ đường dẫn
        if img.ndim == 3:  # Nếu ảnh là ảnh màu, chuyển sang không gian màu xám
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img

        # Chuyển ảnh xám sang nhị phân bằng ngưỡng Otsu
        _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Lưu ảnh nhị phân
        binary_image_path = os.path.join(output_binary_folder, f'binary_{i}')
        cv2.imwrite(binary_image_path, img_binary)

        # Trích xuất đặc trưng Hu Moments từ ảnh nhị phân
        hu_features = extract_feature(img_binary)
        features.append(hu_features)
        labels.append(label)

    return features, labels

def save_to_csv(features, labels, file_name):
    # Lưu các đặc trưng và nhãn vào tệp CSV
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(file_name, index=False)

def z_score_standardization(input_csv, output_csv):
    # Tải tệp CSV vào DataFrame
    df = pd.read_csv(input_csv)

    # Áp dụng chuẩn hóa Z-score cho các cột đặc trưng (giả sử cột cuối cùng là nhãn)
    feature_columns = df.columns[:-1]  # Loại bỏ cột nhãn
    df[feature_columns] = df[feature_columns].apply(zscore)

    # Lưu dữ liệu đã chuẩn hóa vào tệp CSV mới
    df.to_csv(output_csv, index=False)

# Đường dẫn ảnh đầu vào và lưu lại ảnh nhị phân
output_binary_folder = r'E:/Downloads/DATA/Hu/binary_images'  # Thư mục lưu ảnh xám
la_chi, nhanlachi = extract_hu_features_from_folder(       r"E:/Downloads/DATA/LEAF/THAIDUCTOAN/la_chi", 1, output_binary_folder=output_binary_folder)                  #lachi
la_cham, nhanlacham = extract_hu_features_from_folder(     r"E:/Downloads/DATA/LEAF/TRANTHANHKHOA/la_cham", 2, output_binary_folder=output_binary_folder)     #lacham
la_phong, nhanlaphong = extract_hu_features_from_folder(   r"E:/Downloads/DATA/LEAF/MAIDUCKHIEM/la_phong", 3, output_binary_folder=output_binary_folder)                    #laphong
la_tao, nhanlatao = extract_hu_features_from_folder(       r"E:/Downloads/DATA/LEAF/NGOHUUMINH/la tao", 4, output_binary_folder=output_binary_folder)                                       #la tao
rau_muong, nhanraumuong = extract_hu_features_from_folder( r"E:/Downloads/DATA/LEAF/NGUYENTHANHLAN/rau muong", 5, output_binary_folder=output_binary_folder)                                #rau muong


# Lưu vào file csv
features = la_chi + la_cham + la_phong + la_tao + rau_muong
labels = nhanlachi + nhanlacham + nhanlaphong + nhanlatao + nhanraumuong
save_to_csv(features, labels, r'E:/Downloads/DATA/Hu/hutest_hsv/HUnhom11.csv')

# Chuẩn hóa dữ liệu Hu Moments
input_csv_path =    r'E:/Downloads/DATA/Hu/hutest_hsv/HUnhom11.csv'
output_csv_path =   r'E:/Downloads/DATA/Hu/hutest_hsv/HUnhom11Std.csv'
z_score_standardization(input_csv_path, output_csv_path)
