from sklearn.metrics import f1_score
import streamlit as st
import pandas as pd
from pickle import load
from sklearn.preprocessing import LabelEncoder
import pickle 
import os
import pandas as pd
import sys

# sys.path.append(r'C:\Users\luyen\KLTN\fake\imbDRL-master')
from imbDRL.agents.ddqn import TrainDDQN

from imbDRL.metrics import (classification_metrics, network_predictions,
                            plot_confusion_matrix, plot_pr_curve,
                            plot_roc_curve)
from imbDRL.utils import rounded_dict
from sklearn.model_selection import train_test_split
import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU on structured data

st.set_page_config(page_title="Home Page",layout="wide")
st.title("Non-Performing Customer Prediction")
st.image("""https://th.bing.com/th/id/R.1e470560b960cb6693e8e77a30f321b7?rik=ZX2Z8oUcGWZ8Tw&riu=http%3a%2f%2fthicao.com%2fwp-content%2fuploads%2f2019%2f03%2fthiet-ke-logo-nhan-dien-thuong-hieu-f88-1.jpg&ehk=TWX1hBbDVKzNVECDzITK22L1TnvedmJbIXB%2fYcw2lgg%3d&risl=&pid=ImgRaw&r=0""", width=620)
st.header('Enter the characteristics of the customer:')

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)

min_class = [1]  # Minority classes, same setup as in original paper
maj_class = [0]  # Majority classes

sc = joblib.load('w/std_scaler.save')

@st.cache_data
def get_data():
    df = pd.read_parquet("final_preprocess.parquet",engine = 'fastparquet')
    fp_model = "models/20230603_003037.pkl"
    return df, fp_model

df,  fp_model= get_data()

df = df[['AREA', 'CATEGORYNAME', 'PAPERTYPE', 'PROVINCE_SHOP',
       'DESCRIPTION_x', 'KENH',
       'LOAI_KHACH_HANG', 'LOAI_HINH_CU_TRU', 'WORKPLACE_CODE',
       'MARITAL_ID', 'INDUSTRY_NM', 'JOB_NM', 'STATUS',
       'NUMBER_OF_CHILD', 'LOAN_PURPOSE_NAME', 'RESIDENCE_TIME', 'DISTANCE',
       'IS_BAD_DEBT', 'PACKAGE_CODE_F88', 'IS_CUSTOMER_NEW_0_1', 'INCOME', 'MONEY_APPRAISAL', 'AGE']]

def NPL(value):
    if value == 1:
        return "Bad Debt (Non-Performing Loan)"
    else:
        return "Good Debt"

with st.form(key='my_form'):
    
    for i in ['AREA', 'CATEGORYNAME', 'PAPERTYPE', 'PROVINCE_SHOP',
        'DESCRIPTION_x', 'KENH',
        'LOAI_KHACH_HANG', 'LOAI_HINH_CU_TRU','WORKPLACE_CODE',
        'INDUSTRY_NM', 'JOB_NM',
        'NUMBER_OF_CHILD', 'LOAN_PURPOSE_NAME', 'RESIDENCE_TIME',
        'IS_BAD_DEBT', 'PACKAGE_CODE_F88', 'IS_CUSTOMER_NEW_0_1']:
        option = st.selectbox(i, options = df[i].unique(), key = i)
    option = st.selectbox("MARITAL", options = df["MARITAL_ID"].unique(), key = "MARITAL_ID")
    option = st.number_input('INCOME:', format="%.f", key = 'INCOME')
    option  = st.number_input('MONEY_APPRAISAL:', format="%.f", key = 'MONEY_APPRAISAL')
    option = st.number_input('AGE:', format="%.f", key = 'AGE')
    option = st.number_input('DISTANCE:', format="%.f", key = 'DISTANCE')

    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        X_test = pd.DataFrame([[st.session_state[i] for i in ['AREA', 'CATEGORYNAME', 'PAPERTYPE', 'PROVINCE_SHOP',
       'DESCRIPTION_x', 'KENH',
       'LOAI_KHACH_HANG', 'LOAI_HINH_CU_TRU', 'WORKPLACE_CODE',
       'MARITAL_ID', 'INDUSTRY_NM', 'JOB_NM',
       'NUMBER_OF_CHILD', 'LOAN_PURPOSE_NAME', 'RESIDENCE_TIME', 'DISTANCE',
       'IS_BAD_DEBT', 'PACKAGE_CODE_F88', 'IS_CUSTOMER_NEW_0_1', 'INCOME', 'MONEY_APPRAISAL', 'AGE']]], 
        columns=['AREA', 'CATEGORYNAME', 'PAPERTYPE', 'PROVINCE_SHOP',
       'DESCRIPTION_x', 'KENH',
       'LOAI_KHACH_HANG', 'LOAI_HINH_CU_TRU', 'WORKPLACE_CODE',
       'MARITAL_ID', 'INDUSTRY_NM', 'JOB_NM',
       'NUMBER_OF_CHILD', 'LOAN_PURPOSE_NAME', 'RESIDENCE_TIME', 'DISTANCE',
       'IS_BAD_DEBT', 'PACKAGE_CODE_F88', 'IS_CUSTOMER_NEW_0_1', 'INCOME', 'MONEY_APPRAISAL', 'AGE'])
        X_test["STATUS"] = 1

        pkl_des = open('w/DESCRIPTION_x.pkl', 'rb')
        le_des = pickle.load(pkl_des) 
        pkl_des.close()
        X_test['DESCRIPTION_x'] = le_des.transform(X_test['DESCRIPTION_x'])

        pkl_lp = open('w/LOAN_PURPOSE_NAME.pkl', 'rb')
        le_lp = pickle.load(pkl_lp) 
        pkl_lp.close()
        X_test['LOAN_PURPOSE_NAME'] = le_lp.transform(X_test['LOAN_PURPOSE_NAME'])

        pkl_in = open('w/INDUSTRY_NM.pkl', 'rb')
        le_in = pickle.load(pkl_in) 
        pkl_in.close()
        X_test['INDUSTRY_NM'] = le_in.transform(X_test['INDUSTRY_NM'])

        pkl_ps = open('w/PROVINCE_SHOP.pkl', 'rb')
        le_ps = pickle.load(pkl_ps) 
        pkl_ps.close()
        X_test['PROVINCE_SHOP'] = le_ps.transform(X_test['PROVINCE_SHOP'])

        dummies_frame = ['PROVINCE_SHOP', 'DESCRIPTION_x', 'WORKPLACE_CODE', 'INDUSTRY_NM',
       'STATUS', 'NUMBER_OF_CHILD', 'LOAN_PURPOSE_NAME', 'DISTANCE',
       'IS_BAD_DEBT', 'PACKAGE_CODE_F88', 'IS_CUSTOMER_NEW_0_1', 'INCOME',
       'MONEY_APPRAISAL', 'AGE', 'AREA_Miền Bắc', 'AREA_Miền Nam',
       'AREA_Miền Trung', 'CATEGORYNAME_Đăng ký xe máy',
       'CATEGORYNAME_Đăng ký Ô tô', 'PAPERTYPE_KHÁC', 'PAPERTYPE_KT1',
       'PAPERTYPE_KT3', 'KENH_AGENT E2E', 'KENH_Agent', 'KENH_DRS', 'KENH_E2E',
       'KENH_KDML', 'KENH_Recall E2E', 'LOAI_KHACH_HANG_Cá nhân',
       'LOAI_KHACH_HANG_Tổ chức', 'LOAI_HINH_CU_TRU_KT1',
       'LOAI_HINH_CU_TRU_KT3', 'LOAI_HINH_CU_TRU_other', 'MARITAL_ID_20021.0',
       'MARITAL_ID_20022.0', 'MARITAL_ID_20023.0', 'MARITAL_ID_20024.0',
       'MARITAL_ID_20025.0', 'MARITAL_ID_UnKnown', 'JOB_NM_Bác sĩ/Kỹ sư',
       'JOB_NM_Bán hàng', 'JOB_NM_Bảo vệ', 'JOB_NM_Công nhân/Thợ',
       'JOB_NM_Giúp việc/Tạp vụ/ Vệ sinh', 'JOB_NM_Không xác định',
       'JOB_NM_Nhân viên văn phòng', 'JOB_NM_Quản lý/Chủ Doanh nghiệp',
       'JOB_NM_Quân đội/Công An/Viên chức nhà nước', 'JOB_NM_Sinh viên',
       'JOB_NM_Tài xế công nghệ', 'JOB_NM_Tài xế công nghệ/Shipper',
       'JOB_NM_Thợ làm tóc/trang điểm/Spa', 'JOB_NM_Thợ may',
       'JOB_NM_Thợ xây/ sửa chữa/cơ khí…', 'JOB_NM_Tiểu thương buôn bán',
       'JOB_NM_Đầu bếp/Phụ bếp/Bồi bàn/Phục vụ',
       'RESIDENCE_TIME_1 năm', 'RESIDENCE_TIME_1 –> 3 năm',
       'RESIDENCE_TIME_3 –> 5 năm', 'RESIDENCE_TIME_> 5 năm',
       'RESIDENCE_TIME_Không chia sẻ', 'RESIDENCE_TIME_Không xác định']
        X_test = X_test.reindex(columns = dummies_frame, fill_value=0)
        X_test = X_test[['PROVINCE_SHOP', 'DESCRIPTION_x', 'WORKPLACE_CODE', 'INDUSTRY_NM',
       'STATUS', 'NUMBER_OF_CHILD', 'LOAN_PURPOSE_NAME', 'DISTANCE',
       'IS_BAD_DEBT', 'PACKAGE_CODE_F88', 'IS_CUSTOMER_NEW_0_1', 'INCOME',
       'MONEY_APPRAISAL', 'AGE', 'AREA_Miền Bắc', 'AREA_Miền Nam',
       'AREA_Miền Trung', 'CATEGORYNAME_Đăng ký xe máy',
       'CATEGORYNAME_Đăng ký Ô tô', 'PAPERTYPE_KHÁC', 'PAPERTYPE_KT1',
       'PAPERTYPE_KT3', 'KENH_AGENT E2E', 'KENH_Agent', 'KENH_DRS', 'KENH_E2E',
       'KENH_KDML', 'KENH_Recall E2E', 'LOAI_KHACH_HANG_Cá nhân',
       'LOAI_KHACH_HANG_Tổ chức', 'LOAI_HINH_CU_TRU_KT1',
       'LOAI_HINH_CU_TRU_KT3', 'LOAI_HINH_CU_TRU_other', 'MARITAL_ID_20021.0',
       'MARITAL_ID_20022.0', 'MARITAL_ID_20023.0', 'MARITAL_ID_20024.0',
       'MARITAL_ID_20025.0', 'MARITAL_ID_UnKnown', 'JOB_NM_Bác sĩ/Kỹ sư',
       'JOB_NM_Bán hàng', 'JOB_NM_Bảo vệ', 'JOB_NM_Công nhân/Thợ',
       'JOB_NM_Giúp việc/Tạp vụ/ Vệ sinh', 'JOB_NM_Không xác định',
       'JOB_NM_Nhân viên văn phòng', 'JOB_NM_Quản lý/Chủ Doanh nghiệp',
       'JOB_NM_Quân đội/Công An/Viên chức nhà nước', 'JOB_NM_Sinh viên',
       'JOB_NM_Tài xế công nghệ', 'JOB_NM_Tài xế công nghệ/Shipper',
       'JOB_NM_Thợ làm tóc/trang điểm/Spa', 'JOB_NM_Thợ may',
       'JOB_NM_Thợ xây/ sửa chữa/cơ khí…', 'JOB_NM_Tiểu thương buôn bán',
       'JOB_NM_Đầu bếp/Phụ bếp/Bồi bàn/Phục vụ',
       'RESIDENCE_TIME_1 năm', 'RESIDENCE_TIME_1 –> 3 năm',
       'RESIDENCE_TIME_3 –> 5 năm', 'RESIDENCE_TIME_> 5 năm',
       'RESIDENCE_TIME_Không chia sẻ', 'RESIDENCE_TIME_Không xác định']]
        cols = X_test.columns

        X_test = pd.DataFrame(sc.transform(X_test), columns=cols)
        X_test = X_test.to_numpy()
        network = TrainDDQN.load_network(fp_model)
        y_pred_test = network_predictions(network, X_test)
        st.write("Result: ",NPL(y_pred_test[0]))


