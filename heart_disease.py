# Import Required Libraries
import pandas as pd
import numpy as np
import pickle
import time
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

# Page Configuration
st.set_page_config(layout="wide", page_title="Capstone Project DQLab", page_icon=":heart:")
st.sidebar.title("Navigation")
nav = st.sidebar.selectbox("Go to", ("Home", "About Dataset", "Exploratory Data Analysis","Modelling", "Prediction", "About Me"))

# Dataset Page
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
df = pd.read_csv(url)

# Function Heart Disease Prediction
def heart():
    st.write("""
    This app predicts the **Heart Disease**
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1 
            age = st.sidebar.slider("Usia", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
    
    input_df = user_input_features()
    img = Image.open("heart-disease2.jpg")
    st.image(img, width=500)
    if st.sidebar.button('Predict'):
        df = input_df
        st.write(df)
        with open("model_heart_disease.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)        
        result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")

# Home Page
if nav == "Home":
    st.title("Capstone Project Heart Disease DQLab")
    st.write('''
    **Machine Learning & AI Track**
    
    Halo, perkenalkan nama saya [Aristo Budiman](https://github.com/AristoBudiman). 
    Saya adalah seorang mahasiswa Universitas Sebelas Maret (UNS) jurusan informatika semester 3.
    Saya mengikuti kelas Machine Learning & AI di DQLab Academy. Ini adalah capstone project saya tentang prediksi kesehatan jantung dengan machine learning.
    ''')
    st.image('heart-disease1.jpg',
             width=500)

    st.write('''
    **Project Overview**
    
    Cardiovascular disease (CVDs) atau penyakit jantung merupakan penyebab kematian nomor satu secara global dengan 17,9 juta kasus kematian setiap tahunnya. 
    Penyakit jantung disebabkan oleh hipertensi, obesitas, dan gaya hidup yang tidak sehat. Deteksi dini penyakit jantung perlu dilakukan pada kelompok risiko tinggi agar dapat segera mendapatkan penanganan dan pencegahan. 
    Sehingga tujuan bisnis yang ingin dicapai yaitu membentuk model prediksi penyakit jantung pada pasien berdasarkan feature-feature yang ada untuk membantu para dokter melakukan diagnosa secara tepat dan akurat. 
    Harapannya agar penyakit jantung dapat ditangani lebih awal. Dengan demikian, diharapkan juga angka kematian akibat penyakit jantung dapat turun. Dataset yang digunakan adalah dataset penyakit jantung dari [UCI 
    Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    ''')

    st.write('''
    **Problem Statement**
    
    Masalah yang ingin di selesaikan adalah melakukan diagnosa pasien penderita penyakit jantung secara tepat dan akurat. 
    Perlu dilakukan analisis faktor-faktor penyebab dan gejala penyakit jantung pada pasien.
    ''')

    st.write('''
    **Project Objective**
    
    Tujuan dari Capstone adalah melakukan data preprocessing termasuk Exploratory Data Analysis untuk menggali insight dari data pasien penderita penyakit jantung hingga proses feature selection dan dimensionality reduction. Hasil akhir yang ingin dicapai yaitu mendapatkan insight data penderita penyakit jantung dan data yang siap untuk dimodelkan pada tahap selanjutnya.
    ''')

elif nav == 'About Dataset':
    st.title("About Dataset")
    st.write('''

    Dataset yang digunakan adalah dataset penyakit jantung dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    Dataset yang digunakan ini berasal dari tahun 1988 dan terdiri dari empat database: Cleveland, Hungaria, Swiss, dan Long Beach V.
    Dataset heart disease terdiri dari 1025 baris data dan 13 atribut + 1 target. Kolom target adalah kolom **target** yang menunjukkan apakah seseorang
    memiliki penyakit jantung atau tidak. Jika memiliki penyakit jantung, maka nilai kolom **target** adalah 1, jika tidak
    memiliki penyakit jantung, maka nilai kolom **target** adalah 0. Dataset ini memiliki 14 kolom yaitu:
    
    1. **age** : usia dalam tahun
    2. **sex** : jenis kelamin 
        - 1: laki-laki
        - 0: perempuan
    3. **cp** : tipe nyeri dada
        - 1: typical angina
        - 2: atypical angina
        - 3: non-anginal pain
        - 4: asymptomatic
    4. **trestbps** : tekanan darah istirahat dalam mm Hg
    5. **chol** : serum kolestoral dalam mg/dl
    6. **fbs** : gula darah puasa > 120 mg/dl 
        - 1: true
        - 0: false
    7. **restecg** : hasil elektrokardiografi istirahat
        - 0: normal
        - 1: adanya kelainan gelombang ST-T
        - 2: hipertrofi ventrikel kiri
    8. **thalach** : detak jantung maksimum yang dicapai dalam bpm
    9. **exang** : 
        - 1: dipicu oleh aktivitas olahraga
        - 0: tidak dipicu oleh aktivitas olahraga
    10. **oldpeak** : ST depression yang disebabkan oleh olahraga relatif terhadap istirahat
    11. **slope** : kemiringan segmen ST latihan puncak
        - 1: naik
        - 2: datar
        - 3: turun
    12. **ca** : jumlah pembuluh darah utama (0-3) yang terlihat pada pemeriksaan flourosopi
    13. **thal** : 
        - 1: kondisi normal.
        - 2: adanya defek tetap pada thalassemia
        - 3: adanya defek yang dapat dipulihkan pada thalassemia
    14. **target** : 
        - 1: memiliki penyakit jantung
        - 0: tidak memiliki penyakit jantung
    ''')

    # show dataset
    st.write('''
    **Show Dataset**
    ''')
    st.dataframe(df.head())

    # show dataset shape
    st.write(f'''**Dataset Shape:** {df.shape}''')

    # show dataset description
    st.write('''
    **Dataset Description**
    ''')
    st.dataframe(df.describe())

    # show dataset count visualization
    st.write('''
    **Dataset Count Visualization**
    ''')
    views = st.selectbox("Select Visualization", ("", "Target", "Age"))
    if views == "Target":
        st.bar_chart(df.target.value_counts())
        st.write('''
        **Target** adalah kolom yang menunjukkan apakah seseorang memiliki penyakit jantung atau tidak. Jika memiliki penyakit
        jantung, maka nilai kolom **target** adalah 1, jika tidak memiliki penyakit jantung, maka nilai kolom **target** adalah 0.
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung lebih banyak daripada
        yang tidak memiliki penyakit jantung sejumlah 526 orang dibandingkan 499 orang.
        ''')
    elif views == "Age":
        st.bar_chart(df['age'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada usia 58 tahun sebanyak 68 orang. Sedangkan jumlah orang yang tidak memiliki penyakit jantung paling banyak berada
        rentang 74-76 tahun sebanyak 9 orang.''')

elif nav == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.write('''
    **Data Cleaning**
    
    Pada tahap ini, dilakukan pengecekan terhadap data apakah terdapat data yang kosong atau tidak. Jika terdapat data yang
    kosong, maka data tersebut akan dihapus.
    ''')
    st.write('''
    Informasi yang akan kita gali adalah feature pada kesalahan penulisan:
    1. Feature **CA**: Memiliki 5 nilai dari rentang 0-4, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
    2. Feature **thal**: Memiliki 4 nilai dari rentang 0-3, maka dari itu nulai 0 diubah menjadi NaN (karena seharusnya tidak ada)
    ''')
    views = st.radio("Show Data", ("CA", "Thal"))
    if views == "CA":
        st.write('''
        **Feature CA**
        
        Feature CA memiliki 5 nilai dari rentang 0-4, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')
        st.dataframe(df.ca.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**
        ''')
        st.dataframe(df.ca.replace(0, np.nan).value_counts().to_frame().transpose())
    elif views == "Thal":
        st.write('''
        **Feature Thal**
        
        Feature Thal memiliki 4 nilai dari rentang 0-3, maka dari itu nulai 0 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')
        st.dataframe(df.thal.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**
        ''')
        st.dataframe(df.thal.replace(0, np.nan).value_counts().to_frame().transpose())

    st.write('''
    **Handling Outlier and Duplicate Data**
    
    Pada tahap ini, dilakukan pengecekan terhadap data outlier dan data duplikat.
    ''')

    st.image('BeforeCheckOutlier.png', width=500)
    st.image('AfterCheckOutlier.png', width=500)

    st.write('''
    **Korelasi antar Variabel**
    
    Pada tahap ini, dilakukan pengecekan terhadap korelasi variabel dengan target untuk mengeliminasi variabel yang korelasinya dengan target rendah.
    Variabel yang dipilih yaitu :'cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', dan 'age' untuk dianalisa lebih lanjut.
    ''')

    st.image('KorelasiVariabel.png', width=1000)

    st.write('''
    **EDA lengkap bisa dilihat di** [LinkGoogleCollab](https://colab.research.google.com/drive/1fPkYofs9bvORDXJrFsvpk-hHkGi5eypi?usp=sharing)
    
    ''')

elif nav == "Modelling":
    st.header("Modelling")
    var = st.select_slider("Select Model", ("Before Tuning", "After Tuning", "ROC-AUC", "Kesimpulan"))
    if var == "Before Tuning":
        accuracy_score = {
            'Logistic Regression': 0.82,
            'Decision Tree': 0.74,
            'Random Forest': 0.84,
            'MLP Classifier': 0.81,
        }
        st.write('''
        **Model Before Tuning**
        
        Berikut adalah hasil akurasi dari model sebelum dilakukan tuning.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model sebelum dilakukan tuning, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah Random Forest dengan akurasi 0.84.
        ''')

    elif var == "After Tuning":
        accuracy_score = {
            'Logistic Regression': 0.82,
            'Decision Tree': 0.81,
            'Random Forest': 0.86,
            'MLP Classifier': 0.84,
        }
        st.write('''
        **Model After Tuning**
        
        Berikut adalah hasil akurasi dari model setelah dilakukan tuning. Dapat dilihat bahwa model dengan akurasi tertinggi
        adalah Random Forest dengan akurasi 0.86.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model setelah dilakukan tuning, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah MLP Classifier dengan akurasi 0.89.
        ''')

    elif var == "ROC-AUC":
        roc_auc_score = {
            'Logistic Regression': 0.88,
            'Decision Tree': 0.88,
            'Random Forest': 0.90,
            'MLP Classifier': 0.90,
        }
        st.write("**ROC-AUC Scores**")
        st.dataframe(pd.DataFrame(roc_auc_score.items(), columns=['Model', 'ROC-AUC Score']))

    elif var == "Kesimpulan":
        best_threshold_score = {
            'Logistic Regression': 0.47,
            'Decision Tree': 0.61,
            'Random Forest': 0.56,
            'MLP Classifier': 0.54,
        }
        st.write("**Best Threshold**")
        st.dataframe(pd.DataFrame(best_threshold_score.items(), columns=['Model', 'Best Threshold']))

        st.write('''
        **Kesimpulan**
        
        Berdasarkan hasil akurasi dari model sebelum dan setelah dilakukan tuning, dapat disimpulkan bahwa model dengan
        akurasi tertinggi adalah Random Forest yang akurasi after tunning tertinggi yaitu 0.86 dan best threshold tidak jauh di atas 0.50 karena untuk memprediksi true positive rate.
        Anda dapat mendownload model di link berikut ini [Download Model](https://drive.google.com/file/d/1X9nKSY3taMwxbBYM18KgZVnTqoi6PuW_/view?usp=drive_link).
        ''')

elif nav == 'Prediction':
    st.header("Heart Disease Prediction")
    heart()

elif nav == "About Me":
    st.title("About Me")
    st.image('ProfilePicture.jpg', width=200)
    st.write('''
    **Aristo Budiman**
    
    Saya adalah seorang mahasiswa Universitas Sebelas Maret (UNS) jurusan informatika semester 3.
    Saya mengikuti kelas Machine Learning & AI di DQLab Academy. Ini adalah capstone project saya tentang prediksi kesehatan jantung dengan machine learning.
    ''')
    st.write('''
    **Contact Me**
    
    - [LinkedIn](https://www.linkedin.com/in/aristobudiman)
    - [Github](https://github.com/AristoBudiman)
    ''')

