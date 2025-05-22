import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the categories
categories = ['Politics', 'Business', 'Technology', 'Health', 'Sports', 'Entertainment', 'Education']

# Define characteristic keywords for each category
category_keywords = {
    'Politics': ['thu_tuong', 'dai_bieu', 'nghi_quyet', 'quoc_hoi', 'chinh_phu', 'nha_nuoc', 'dang', 'bo_truong', 'lanh_dao', 'chinh_sach', 'hiep_dinh', 'ngoai_giao', 'luat', 'quyen_luc', 'bau_cu'],
    'Business': ['thi_truong', 'doanh_nghiep', 'lai_suat', 'kinh_doanh', 'co_phieu', 'ngan_hang', 'dau_tu', 'chung_khoan', 'loi_nhuan', 'thuong_mai', 'xuat_khau', 'nhap_khau', 'gdp', 'tai_chinh', 'tien_te'],
    'Technology': ['phan_mem', 'ai', 'cong_nghe', 'thiet_bi', 'app', 'mang', 'internet', 'smartphone', 'may_tinh', 'robot', 'hack', 'web', 'game', 'facebook', 'google'],
    'Health': ['bac_si', 'vaccine', 'y_te', 'benh_vien', 'suc_khoe', 'dich_benh', 'thuoc', 'covid', 'dieu_tri', 'phong_benh', 'khau_trang', 'phau_thuat', 'dinh_duong', 'tap_luyen', 'bao_hiem'],
    'Sports': ['vo_dich', 'cau_thu', 'bong_da', 'tran_dau', 'the_thao', 'giai_dau', 'huy_chuong', 'olympic', 'clb', 'hlv', 'cup', 'san_van_dong', 'ghi_ban', 'sea_games', 'doi_tuyen'],
    'Entertainment': ['truyen_hinh', 'dien_vien', 'phim', 'ca_si', 'am_nhac', 'show', 'concert', 'thoi_trang', 'mv', 'game_show', 'scandal', 'fan', 'sao', 'rap', 'tv_show'],
    'Education': ['hoc_sinh', 'dai_hoc', 'truong_hoc', 'giao_vien', 'thi', 'giao_duc', 'hoc_bong', 'mon_hoc', 'tot_nghiep', 'diem', 'lop', 'thu_khoa', 'nghien_cuu', 'sinh_vien', 'dao_tao']
}
import pandas as pd
import numpy as np
import random

# 1. ĐỊNH NGHĨA CÁC THAM SỐ VÀ TỪ KHÓA
NUM_SAMPLES = 1000
CATEGORIES = {
    "Politics": ["chinh_phu", "quoc_hoi", "bau_cu", "luat", "nghi_quyet", "chinh_sach", "dang", "bo_truong", "thu_tuong", "chu_tich", "dai_bieu", "ngoai_giao", "hiep_dinh", "hoi_nghi", "lanh_dao"],
    "Business": ["co_phieu", "thi_truong", "gia_vang", "doanh_nghiep", "lam_phat", "ngan_hang", "lai_suat", "khung_hoang", "gdp", "tang_truong", "dau_tu", "xuat_khau", "nhap_khau", "kinh_doanh", "thuong_mai"],
    "Technology": ["ai", "an_ninh_mang", "mang_5g", "startup", "blockchain", "cong_nghe", "may_tinh", "robot", "app", "internet", "dien_thoai", "phan_mem", "thiet_bi", "du_lieu", "thong_minh"],
    "Health": ["vaccine", "benh_vien", "bac_si", "dich_benh", "dinh_duong", "suc_khoe", "thuoc", "tiem_chung", "cham_soc", "phau_thuat", "khong_khi", "bien_chung", "nhiem", "ung_thu", "y_te"],
    "Sports": ["doi_tuyen", "world_cup", "cau_thu", "olympic", "v_league", "bong_da", "the_thao", "vo_dich", "huy_chuong", "giai_dau", "hlv", "the_van_hoi", "tran_dau", "tennis", "bong_ro"],
    "Entertainment": ["phim", "ca_si", "am_nhac", "gameshow", "mv", "nghe_si", "bai_hat", "truyen_hinh", "concert", "thu_hut", "dien_vien", "dao_dien", "liveshow", "san_khau", "idol"],
    "Education": ["dai_hoc", "diem_chuan", "tot_nghiep", "hoc_sinh", "chuong_trinh", "giao_duc", "giao_vien", "truong_hoc", "thi", "mon_hoc", "hoc_phi", "nam_hoc", "thu_khoa", "hoc_bong", "dao_tao"]
}

# Từ khóa chung xuất hiện ở nhiều danh mục
COMMON_KEYWORDS = ["viet_nam", "ha_noi", "tphcm", "nguoi_dan", "thanh_nien", "xa_hoi", "phat_trien", 
                  "thuc_hien", "trien_khai", "hoat_dong", "quoc_te", "thong_tin", "van_de", "cong_dong", "hoc_tap"]

# Thêm các từ khóa chung vào danh sách chung
ALL_KEYWORDS = COMMON_KEYWORDS.copy()
for cat_keywords in CATEGORIES.values():
    ALL_KEYWORDS.extend(cat_keywords)
ALL_KEYWORDS = sorted(list(set(ALL_KEYWORDS)))  # Danh sách duy nhất và đã sắp xếp của tất cả từ khóa

NUM_KEYWORDS_PER_CATEGORY = 15
MISSING_VALUE_PERCENTAGE = 0.05
OVERLAP_FACTOR = 0.3  # Xác suất một từ khóa thuộc danh mục khác sẽ xuất hiện với tần suất cao
NOISE_FACTOR = 0.15   # Xác suất một bài viết sẽ có nhiều nhiễu
MISLABEL_PERCENTAGE = 0.08  # Phần trăm bài viết bị gán nhãn sai

# 2. TẠO DỮ LIỆU
dataset = []
category_list = list(CATEGORIES.keys())

# Phân bổ mẫu gần đều cho các category
samples_per_category = NUM_SAMPLES // len(category_list)
remainder = NUM_SAMPLES % len(category_list)
category_distribution = [samples_per_category] * len(category_list)
for i in range(remainder):
    category_distribution[i] += 1
random.shuffle(category_distribution)  # Xáo trộn để không theo thứ tự

assigned_categories = []
for i, count in enumerate(category_distribution):
    assigned_categories.extend([category_list[i]] * count)
random.shuffle(assigned_categories)  # Xáo trộn thứ tự các mẫu

# Xác định số lượng mẫu sẽ bị gán nhãn sai
num_mislabeled = int(NUM_SAMPLES * MISLABEL_PERCENTAGE)
mislabeled_indices = random.sample(range(NUM_SAMPLES), num_mislabeled)

for i in range(NUM_SAMPLES):
    sample_id = i + 1
    main_category_name = assigned_categories[i]
    
    # Nếu là mẫu bị gán nhãn sai, chọn một danh mục khác làm danh mục thực tế
    actual_category_name = main_category_name
    if i in mislabeled_indices:
        possible_wrong_categories = [c for c in category_list if c != main_category_name]
        actual_category_name = random.choice(possible_wrong_categories)
    
    main_category_keywords = CATEGORIES[actual_category_name]  # Sử dụng từ khóa của danh mục thực tế

    # Khởi tạo tần suất từ khóa bằng 0
    keyword_frequencies = {keyword: 0 for keyword in ALL_KEYWORDS}
    doc_length = 0

    # Xác định nếu mẫu này sẽ có nhiều nhiễu
    is_noisy = random.random() < NOISE_FACTOR

    # Tần suất cho từ khóa CHÍNH
    # Chọn ngẫu nhiên một số từ khóa chính sẽ xuất hiện
    min_main_keywords = 3 if is_noisy else 5
    max_main_keywords = 8 if is_noisy else NUM_KEYWORDS_PER_CATEGORY - 2
    num_main_keywords_to_appear = random.randint(min_main_keywords, max_main_keywords)
    selected_main_keywords = random.sample(main_category_keywords, num_main_keywords_to_appear)
    
    for keyword in selected_main_keywords:
        # Tần suất thấp hơn nếu là bài viết nhiễu
        if is_noisy:
            freq = random.randint(2, 8)
        else:
            freq = random.randint(3, 15)
        keyword_frequencies[keyword] = freq
        doc_length += freq

    # Tần suất cho từ khóa CHUNG
    num_common_keywords = random.randint(2, len(COMMON_KEYWORDS))
    selected_common_keywords = random.sample(COMMON_KEYWORDS, num_common_keywords)
    for keyword in selected_common_keywords:
        freq = random.randint(1, 10)  # Từ khóa chung có thể xuất hiện nhiều
        keyword_frequencies[keyword] = freq
        doc_length += freq

    # Tần suất cho từ khóa PHỤ (có thể có hoặc không, và từ 0-3 chủ đề phụ)
    num_secondary_categories = random.choices([0, 1, 2, 3], weights=[0.2, 0.3, 0.3, 0.2])[0]
    
    # Tránh chọn lại chủ đề chính làm chủ đề phụ
    available_secondary_categories = [cat for cat in category_list if cat != actual_category_name]
    
    if num_secondary_categories > 0 and available_secondary_categories:
        chosen_secondary_names = random.sample(available_secondary_categories, min(num_secondary_categories, len(available_secondary_categories)))
        for secondary_category_name in chosen_secondary_names:
            secondary_category_keywords = CATEGORIES[secondary_category_name]
            
            # Chọn ngẫu nhiên một số từ khóa phụ sẽ xuất hiện
            num_secondary_keywords_to_appear = random.randint(2, 10)  # Tăng số lượng từ khóa phụ
            selected_secondary_keywords = random.sample(secondary_category_keywords, num_secondary_keywords_to_appear)
            
            for keyword in selected_secondary_keywords:
                # Có xác suất một số từ khóa phụ có tần suất cao (gây nhầm lẫn)
                if random.random() < OVERLAP_FACTOR:
                    freq = random.randint(3, 12)  # Tần suất cao gần với từ khóa chính
                else:
                    freq = random.randint(1, 5)  # Tần suất trung bình
                
                # Cập nhật tần suất từ khóa
                if keyword_frequencies[keyword] == 0:
                    keyword_frequencies[keyword] = freq
                else:
                    keyword_frequencies[keyword] += freq
                doc_length += freq

    # Tần suất cho từ khóa KHÔNG LIÊN QUAN (những từ còn lại)
    # Tăng xác suất xuất hiện của từ khóa không liên quan
    random_noise_probability = 0.25 if is_noisy else 0.15
    for keyword in ALL_KEYWORDS:
        if keyword_frequencies[keyword] == 0:  # Nếu từ này chưa được gán tần suất
            if random.random() < random_noise_probability:
                if is_noisy and random.random() < 0.1:  # 10% khả năng nhiễu cao
                    freq = random.randint(1, 6)  # Tần suất cao bất thường cho từ không liên quan
                else:
                    freq = random.randint(1, 3)  # Tần suất thấp đến trung bình
                keyword_frequencies[keyword] = freq
                doc_length += freq

    # Thêm biến động tự nhiên cho doc_length (có thể không khớp chính xác với tổng từ khóa)
    # Mô phỏng lỗi đếm hoặc từ khóa không được tính
    doc_length_error = random.randint(-5, 5)
    doc_length = max(1, doc_length + doc_length_error)  # Đảm bảo doc_length ít nhất là 1

    # Sắp xếp lại dictionary tần suất từ khóa theo thứ tự của ALL_KEYWORDS
    ordered_frequencies = [keyword_frequencies[kw] for kw in ALL_KEYWORDS]

    # Sử dụng main_category_name làm nhãn (có thể khác với actual_category_name nếu bị gán nhãn sai)
    row = [sample_id] + ordered_frequencies + [doc_length, main_category_name]
    dataset.append(row)

# 3. TẠO DATAFRAME VÀ THÊM GIÁ TRỊ THIẾU
columns = ['id'] + ALL_KEYWORDS + ['doc_length', 'category']
df = pd.DataFrame(dataset, columns=columns)

# Thêm giá trị thiếu (NaN) vào khoảng 5% các ô tần suất từ khóa
num_total_keyword_cells = NUM_SAMPLES * len(ALL_KEYWORDS)
num_missing_values = int(num_total_keyword_cells * MISSING_VALUE_PERCENTAGE)

# Chọn ngẫu nhiên các ô để làm giá trị thiếu
missing_rows = np.random.randint(0, NUM_SAMPLES, size=num_missing_values)
# Chọn ngẫu nhiên các cột từ khóa (tránh cột 'id', 'doc_length', 'category')
keyword_column_indices = np.random.randint(1, 1 + len(ALL_KEYWORDS), size=num_missing_values)

for r, c_idx in zip(missing_rows, keyword_column_indices):
    col_name = df.columns[c_idx]
    
    # Làm giả dữ liệu hơn - không điều chỉnh doc_length khi có giá trị NaN
    # Điều này tạo ra sự không nhất quán giữa doc_length và tổng tần suất
    df.iat[r, c_idx] = np.nan

# 4. LƯU RA FILE CSVimport pandas as pd
import numpy as np
import random

# 1. ĐỊNH NGHĨA CÁC THAM SỐ VÀ TỪ KHÓA
NUM_SAMPLES = 1000
CATEGORIES = {
    "Politics": ["chinh_phu", "quoc_hoi", "bau_cu", "luat", "nghi_quyet", "chinh_sach", "dang", "bo_truong", "thu_tuong", "chu_tich", "dai_bieu", "ngoai_giao", "hiep_dinh", "hoi_nghi", "lanh_dao"],
    "Business": ["co_phieu", "thi_truong", "gia_vang", "doanh_nghiep", "lam_phat", "ngan_hang", "lai_suat", "khung_hoang", "gdp", "tang_truong", "dau_tu", "xuat_khau", "nhap_khau", "kinh_doanh", "thuong_mai"],
    "Technology": ["ai", "an_ninh_mang", "mang_5g", "startup", "blockchain", "cong_nghe", "may_tinh", "robot", "app", "internet", "dien_thoai", "phan_mem", "thiet_bi", "du_lieu", "thong_minh"],
    "Health": ["vaccine", "benh_vien", "bac_si", "dich_benh", "dinh_duong", "suc_khoe", "thuoc", "tiem_chung", "cham_soc", "phau_thuat", "khong_khi", "bien_chung", "nhiem", "ung_thu", "y_te"],
    "Sports": ["doi_tuyen", "world_cup", "cau_thu", "olympic", "v_league", "bong_da", "the_thao", "vo_dich", "huy_chuong", "giai_dau", "hlv", "the_van_hoi", "tran_dau", "tennis", "bong_ro"],
    "Entertainment": ["phim", "ca_si", "am_nhac", "gameshow", "mv", "nghe_si", "bai_hat", "truyen_hinh", "concert", "thu_hut", "dien_vien", "dao_dien", "liveshow", "san_khau", "idol"],
    "Education": ["dai_hoc", "diem_chuan", "tot_nghiep", "hoc_sinh", "chuong_trinh", "giao_duc", "giao_vien", "truong_hoc", "thi", "mon_hoc", "hoc_phi", "nam_hoc", "thu_khoa", "hoc_bong", "dao_tao"]
}

# Từ khóa chung xuất hiện ở nhiều danh mục
COMMON_KEYWORDS = ["viet_nam", "ha_noi", "tphcm", "nguoi_dan", "thanh_nien", "xa_hoi", "phat_trien", 
                  "thuc_hien", "trien_khai", "hoat_dong", "quoc_te", "thong_tin", "van_de", "cong_dong", "hoc_tap"]

# Thêm các từ khóa chung vào danh sách chung
ALL_KEYWORDS = COMMON_KEYWORDS.copy()
for cat_keywords in CATEGORIES.values():
    ALL_KEYWORDS.extend(cat_keywords)
ALL_KEYWORDS = sorted(list(set(ALL_KEYWORDS)))  # Danh sách duy nhất và đã sắp xếp của tất cả từ khóa

NUM_KEYWORDS_PER_CATEGORY = 15
MISSING_VALUE_PERCENTAGE = 0.05
OVERLAP_FACTOR = 0.3  # Xác suất một từ khóa thuộc danh mục khác sẽ xuất hiện với tần suất cao
NOISE_FACTOR = 0.15   # Xác suất một bài viết sẽ có nhiều nhiễu
MISLABEL_PERCENTAGE = 0.08  # Phần trăm bài viết bị gán nhãn sai

# 2. TẠO DỮ LIỆU
dataset = []
category_list = list(CATEGORIES.keys())

# Phân bổ mẫu gần đều cho các category
samples_per_category = NUM_SAMPLES // len(category_list)
remainder = NUM_SAMPLES % len(category_list)
category_distribution = [samples_per_category] * len(category_list)
for i in range(remainder):
    category_distribution[i] += 1
random.shuffle(category_distribution)  # Xáo trộn để không theo thứ tự

assigned_categories = []
for i, count in enumerate(category_distribution):
    assigned_categories.extend([category_list[i]] * count)
random.shuffle(assigned_categories)  # Xáo trộn thứ tự các mẫu

# Xác định số lượng mẫu sẽ bị gán nhãn sai
num_mislabeled = int(NUM_SAMPLES * MISLABEL_PERCENTAGE)
mislabeled_indices = random.sample(range(NUM_SAMPLES), num_mislabeled)

for i in range(NUM_SAMPLES):
    sample_id = i + 1
    main_category_name = assigned_categories[i]
    
    # Nếu là mẫu bị gán nhãn sai, chọn một danh mục khác làm danh mục thực tế
    actual_category_name = main_category_name
    if i in mislabeled_indices:
        possible_wrong_categories = [c for c in category_list if c != main_category_name]
        actual_category_name = random.choice(possible_wrong_categories)
    
    main_category_keywords = CATEGORIES[actual_category_name]  # Sử dụng từ khóa của danh mục thực tế

    # Khởi tạo tần suất từ khóa bằng 0
    keyword_frequencies = {keyword: 0 for keyword in ALL_KEYWORDS}
    doc_length = 0

    # Xác định nếu mẫu này sẽ có nhiều nhiễu
    is_noisy = random.random() < NOISE_FACTOR

    # Tần suất cho từ khóa CHÍNH
    # Chọn ngẫu nhiên một số từ khóa chính sẽ xuất hiện
    min_main_keywords = 3 if is_noisy else 5
    max_main_keywords = 8 if is_noisy else NUM_KEYWORDS_PER_CATEGORY - 2
    num_main_keywords_to_appear = random.randint(min_main_keywords, max_main_keywords)
    selected_main_keywords = random.sample(main_category_keywords, num_main_keywords_to_appear)
    
    for keyword in selected_main_keywords:
        # Tần suất thấp hơn nếu là bài viết nhiễu
        if is_noisy:
            freq = random.randint(2, 8)
        else:
            freq = random.randint(3, 15)
        keyword_frequencies[keyword] = freq
        doc_length += freq

    # Tần suất cho từ khóa CHUNG
    num_common_keywords = random.randint(2, len(COMMON_KEYWORDS))
    selected_common_keywords = random.sample(COMMON_KEYWORDS, num_common_keywords)
    for keyword in selected_common_keywords:
        freq = random.randint(1, 10)  # Từ khóa chung có thể xuất hiện nhiều
        keyword_frequencies[keyword] = freq
        doc_length += freq

    # Tần suất cho từ khóa PHỤ (có thể có hoặc không, và từ 0-3 chủ đề phụ)
    num_secondary_categories = random.choices([0, 1, 2, 3], weights=[0.2, 0.3, 0.3, 0.2])[0]
    
    # Tránh chọn lại chủ đề chính làm chủ đề phụ
    available_secondary_categories = [cat for cat in category_list if cat != actual_category_name]
    
    if num_secondary_categories > 0 and available_secondary_categories:
        chosen_secondary_names = random.sample(available_secondary_categories, min(num_secondary_categories, len(available_secondary_categories)))
        for secondary_category_name in chosen_secondary_names:
            secondary_category_keywords = CATEGORIES[secondary_category_name]
            
            # Chọn ngẫu nhiên một số từ khóa phụ sẽ xuất hiện
            num_secondary_keywords_to_appear = random.randint(2, 10)  # Tăng số lượng từ khóa phụ
            selected_secondary_keywords = random.sample(secondary_category_keywords, num_secondary_keywords_to_appear)
            
            for keyword in selected_secondary_keywords:
                # Có xác suất một số từ khóa phụ có tần suất cao (gây nhầm lẫn)
                if random.random() < OVERLAP_FACTOR:
                    freq = random.randint(3, 12)  # Tần suất cao gần với từ khóa chính
                else:
                    freq = random.randint(1, 5)  # Tần suất trung bình
                
                # Cập nhật tần suất từ khóa
                if keyword_frequencies[keyword] == 0:
                    keyword_frequencies[keyword] = freq
                else:
                    keyword_frequencies[keyword] += freq
                doc_length += freq

    # Tần suất cho từ khóa KHÔNG LIÊN QUAN (những từ còn lại)
    # Tăng xác suất xuất hiện của từ khóa không liên quan
    random_noise_probability = 0.25 if is_noisy else 0.15
    for keyword in ALL_KEYWORDS:
        if keyword_frequencies[keyword] == 0:  # Nếu từ này chưa được gán tần suất
            if random.random() < random_noise_probability:
                if is_noisy and random.random() < 0.1:  # 10% khả năng nhiễu cao
                    freq = random.randint(1, 6)  # Tần suất cao bất thường cho từ không liên quan
                else:
                    freq = random.randint(1, 3)  # Tần suất thấp đến trung bình
                keyword_frequencies[keyword] = freq
                doc_length += freq

    # Thêm biến động tự nhiên cho doc_length (có thể không khớp chính xác với tổng từ khóa)
    # Mô phỏng lỗi đếm hoặc từ khóa không được tính
    doc_length_error = random.randint(-5, 5)
    doc_length = max(1, doc_length + doc_length_error)  # Đảm bảo doc_length ít nhất là 1

    # Sắp xếp lại dictionary tần suất từ khóa theo thứ tự của ALL_KEYWORDS
    ordered_frequencies = [keyword_frequencies[kw] for kw in ALL_KEYWORDS]

    # Sử dụng main_category_name làm nhãn (có thể khác với actual_category_name nếu bị gán nhãn sai)
    row = [sample_id] + ordered_frequencies + [doc_length, main_category_name]
    dataset.append(row)

# 3. TẠO DATAFRAME VÀ THÊM GIÁ TRỊ THIẾU
columns = ['id'] + ALL_KEYWORDS + ['doc_length', 'category']
df = pd.DataFrame(dataset, columns=columns)

# Thêm giá trị thiếu (NaN) vào khoảng 5% các ô tần suất từ khóa
num_total_keyword_cells = NUM_SAMPLES * len(ALL_KEYWORDS)
num_missing_values = int(num_total_keyword_cells * MISSING_VALUE_PERCENTAGE)

# Chọn ngẫu nhiên các ô để làm giá trị thiếu
missing_rows = np.random.randint(0, NUM_SAMPLES, size=num_missing_values)
# Chọn ngẫu nhiên các cột từ khóa (tránh cột 'id', 'doc_length', 'category')
keyword_column_indices = np.random.randint(1, 1 + len(ALL_KEYWORDS), size=num_missing_values)

for r, c_idx in zip(missing_rows, keyword_column_indices):
    col_name = df.columns[c_idx]
    
    # Làm giả dữ liệu hơn - không điều chỉnh doc_length khi có giá trị NaN
    # Điều này tạo ra sự không nhất quán giữa doc_length và tổng tần suất
    df.iat[r, c_idx] = np.nan

# 4. LƯU RA FILE CSV
try:
    df.to_csv('vietnamese_news_dataset.csv', index=False, encoding='utf-8-sig')
    print(f"Bộ dữ liệu đã được tạo và lưu vào 'vietnamese_news_dataset.csv'")
    print(f"Số dòng: {len(df)}")
    print(f"Số cột: {len(df.columns)}")
    print("\nKiểm tra một vài dòng đầu:")
    print(df.head())
    print("\nKiểm tra các cột có giá trị thiếu:")
    missing_info = df.isnull().sum()
    print(missing_info[missing_info > 0])
    print(f"\nSố mẫu có nhãn bị gán sai có chủ đích: {num_mislabeled} ({MISLABEL_PERCENTAGE*100:.1f}%)")
    print(f"Số mẫu có nhiều nhiễu: {int(NUM_SAMPLES * NOISE_FACTOR)} ({NOISE_FACTOR*100:.1f}%)")

except Exception as e:
    print(f"Lỗi khi lưu file: {e}")
try:
    df.to_csv('vietnamese_news_dataset.csv', index=False, encoding='utf-8-sig')
    print(f"Bộ dữ liệu đã được tạo và lưu vào 'vietnamese_news_dataset.csv'")
    print(f"Số dòng: {len(df)}")
    print(f"Số cột: {len(df.columns)}")
    print("\nKiểm tra một vài dòng đầu:")
    print(df.head())
    print("\nKiểm tra các cột có giá trị thiếu:")
    missing_info = df.isnull().sum()
    print(missing_info[missing_info > 0])
    print(f"\nSố mẫu có nhãn bị gán sai có chủ đích: {num_mislabeled} ({MISLABEL_PERCENTAGE*100:.1f}%)")
    print(f"Số mẫu có nhiều nhiễu: {int(NUM_SAMPLES * NOISE_FACTOR)} ({NOISE_FACTOR*100:.1f}%)")

except Exception as e:
    print(f"Lỗi khi lưu file: {e}")
# Create a list of all unique keywords
all_keywords = []
for keywords in category_keywords.values():
    all_keywords.extend(keywords)
all_keywords = list(set(all_keywords))

# Add some general keywords that might appear in any category
general_keywords = ['viet_nam', 'nguoi', 'thang', 'nam', 'the_gioi', 'tin', 'moi', 'phat_trien', 'thong_tin', 'su_kien']
all_keywords.extend(general_keywords)

# Create random additional keywords to reach 105 total
additional_keywords_needed = 105 - len(all_keywords)
if additional_keywords_needed > 0:
    additional_keywords = [f'keyword_{i}' for i in range(1, additional_keywords_needed + 1)]
    all_keywords.extend(additional_keywords)
elif additional_keywords_needed < 0:
    # If we have more than 105 keywords, just keep the first 105
    all_keywords = all_keywords[:105]

# Generate synthetic data
def generate_record(category, idx):
    # Create a dictionary with all keywords initialized to 0
    record = {keyword: 0 for keyword in all_keywords}
    
    # Set higher frequencies for the characteristic keywords of this category
    for keyword in category_keywords[category]:
        # Poisson distribution for keyword counts
        record[keyword] = np.random.poisson(7) if random.random() < 0.9 else 0
    
    # Set lower frequencies for some other keywords
    for cat in categories:
        if cat != category:
            for keyword in category_keywords[cat]:
                if random.random() < 0.1:  # 10% chance of appearing in other categories
                    record[keyword] = np.random.poisson(1)
    
    # Set frequencies for general keywords
    for keyword in general_keywords:
        if random.random() < 0.3:  # 30% chance of appearing
            record[keyword] = np.random.poisson(2)
    
    # Calculate doc_length
    doc_length = sum(record.values())
    
    # Add missing values randomly (about 1% of the data)
    for keyword in all_keywords:
        if random.random() < 0.01:
            record[keyword] = np.nan
    
    # Add ID and category
    record['id'] = idx
    record['category'] = category
    record['doc_length'] = doc_length
    
    return record

# Generate approximately equal number of samples for each category
data = []
idx = 1
samples_per_category = 1000 // len(categories)
remainder = 1000 % len(categories)

for category in categories:
    # Add extra samples to some categories if there's a remainder
    extra = 1 if remainder > 0 else 0
    remainder -= extra
    
    for _ in range(samples_per_category + extra):
        data.append(generate_record(category, idx))
        idx += 1

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the data
df = shuffle(df, random_state=42)

# Reorder columns to put id first, then keywords, then doc_length, then category
column_order = ['id'] + all_keywords + ['doc_length', 'category']
df = df[column_order]

# Export to CSV with UTF-8-sig encoding
df.to_csv('vietnamese_text_classification_data.csv', index=False, encoding='utf-8-sig')

# Print summary statistics
print(f"Generated dataset with {len(df)} samples")
print(f"Number of features: {len(df.columns) - 2}")  # Subtract id and category
print("Category distribution:")
print(df['category'].value_counts())
print("\nSample of the data:")
print(df.head(3))
print("\nData types and missing values:")
print(df.info())
print("\nSummary statistics for keyword frequencies:")
print(df.iloc[:, 1:106].describe())  # Summary of keyword columns