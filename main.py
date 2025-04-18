import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# 数据加载和预处理
def load_and_preprocess():
    # 加载数据集
    file_path = 'D://imputed_data1.csv'  # 使用没有缺失值的数据集
    data = pd.read_csv(file_path)

    # 选择所需的特征
    required_features = [
        'Gender', 'Sleep_night', 'IADL_score', 'Loneliness',
        'Medical Insurance', 'BMI', 'Digestive', 'Martial_status', 'Heart'
    ]

    # 目标变量
    target_col = 'Depression'

    # 特征和目标变量
    X = data[required_features]
    y = data[target_col]

    return X, y


# 加载数据并训练模型
X, y = load_and_preprocess()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# 创建Streamlit界面
st.title('抑郁症风险预测器')

# 用户输入字段
gender = st.selectbox('性别 (0=男, 1=女)', [0, 1])
sleep_night = st.number_input('每晚睡眠小时数', min_value=0, max_value=24, value=7)
IADL_score = st.number_input('IADL得分', min_value=0, max_value=100, value=50)
loneliness = st.selectbox('孤独感 (1=很少, 2=有时, 3=偶尔, 4=经常)', [1, 2, 3, 4])
medical_insurance = st.selectbox('是否有医疗保险 (0=无, 1=有)', [0, 1])
BMI = st.number_input('BMI值', min_value=10.0, max_value=50.0, value=25.0)
digestive = st.selectbox('是否有消化系统疾病 (0=无, 1=有)', [0, 1])
martial_status = st.selectbox('婚姻状况 (1=已婚, 2=单身, 3=离婚/丧偶/分居)', [1, 2, 3])
heart = st.selectbox('是否有心脏病 (0=无, 1=有)', [0, 1])

# 用户输入数据
user_input = np.array(
    [[gender, sleep_night, IADL_score, loneliness, medical_insurance, BMI, digestive, martial_status, heart]])

# 预测按钮
if st.button('预测'):
    # 预测
    prediction = model.predict(user_input)
    if prediction == 1:
        st.write("高风险抑郁症")
    else:
        st.write("低风险抑郁症")
