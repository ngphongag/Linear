import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# train_test_split chia dữ liệu từ data Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
#truyền dữ liệu input vào df
df = pd.read_csv("nhom8.csv", encoding='latin-1')

st.title("Nhóm 8 - Ứng dụng")
st.write("## Xác định khả năng tiếp cận vốn vay")

#upload file để huấn luyện mô hình
uploaded_file = st.file_uploader("Chọn files dữ liệu:", type=['csv'])
#nếu file có dữ liệu thì sẽ thực thi upload dữ liệu vào
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("data.csv", index = False)

# X: giá trị input
X = df.drop(columns=['giatri'])
#y: giá trị target
y = df['giatri']

#test_size=0.2 chia dữ liệu 80 và 20;
#random_state= 12
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 12)

# model LinearRegression khởi tạo thuật toán LinearRegression
model = LinearRegression()

#
model.fit(X_train, y_train)

yhat_test = model.predict(X_test)


score_train=model.score(X_train, y_train)
score_test=model.score(X_test, y_test)


mse=mean_squared_error(y_test, yhat_test)
rmse=mean_squared_error(y_test, yhat_test, squared=False)
mae=mean_absolute_error(y_test, yhat_test)

#xây dựng das
menu = ["Mục tiêu của mô hình","Giới thiệu nhóm", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục Menu', menu)

if choice == 'Mục tiêu của mô hình':    
    st.subheader("Mục tiêu của mô hình")
    st.write("""
    ###### Mô hình được xây dựng để Xác định mức cho vay tối đa.
    """)  
    st.write("""###### Mô hình sử dụng thuật toán LinearRegression""")
    st.image("nhom8.png")
   

elif choice== "Giới thiệu nhóm":
     st.subheader("Thành viên nhóm:")
     st.write("##### 1. Nguyễn Đăng Viết Phong")
     st.write("##### 2. Lê Duy Thịnh")
     st.write("##### 3. Nguyễn Phú Hùng")
     st.write("##### 4. Trần Quang Trung")
     st.write("##### 5. Vũ Xuân Dũng")
     st.write("##### 6. Nguyễn Minh Nhật Tùng")
     st.write("##### 7. Nguyễn Mạnh Tú")
     st.write("##### 8. Đỗ Quang Phát")
     st.write("##### 9. Lê Thị Hằng")
   # đưa hình nhóm vào
   #  st.image("nhom8.png")
               
elif choice == 'Xây dựng mô hình':
    st.subheader("Xây dựng mô hình")
    st.write("##### 1. Hiển thị dữ liệu")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  
    
    st.write("##### 2. Trực quan hóa dữ liệu")
    u=st.text_input('Nhập biến muốn vẽ vào đây')
    fig1 = sns.regplot(data=df, x=u, y='giatri')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model...")
    
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("MSE:"+str(round(mse,2)))
    st.code("RMSE:"+str(round(rmse,2)))
    st.code("MAE:"+str(round(mae,2)))

    
elif choice == 'Sử dụng mô hình để dự báo':
    st.subheader("Sử dụng mô hình để dự báo")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            # st.write(lines.columns)
            flag = True       
    
    # khai báo các biến nhập tay theo đúng tên các trường giá trị DATA
    if type=="Input":        
        git = st.number_input('Khai báo Giá trị cho vay')
        TN = st.number_input('Khai báo thu nhập')
        GTC = st.number_input('Khai báo giá trị tài sản đãm bảo')
        GD = st.number_input('Khai báo số năm đến trường')
        TCH = st.number_input('Khai báo độ tuổi')
        VPCT = st.number_input('Khai báo Nợ khác')
        LS = st.number_input('Khai báo xếp hạng TD')
        lines={'giatri':[git],'TN':[TN],'GTC':[GTC],'GD':[GD],'TCH':[TCH],'VPCT':[VPCT],'LS':[LS]}
        lines=pd.DataFrame(lines)
        st.dataframe(lines)
        flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            X_1 = lines.drop(columns=['giatri'])   
            y_pred_new = model.predict(X_1)       
            st.code("giá trị dự báo: " + str(y_pred_new))
