###################################
# Telco Churn Prediction
###################################

# İş Problemi
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi beklenmektedir.

# Veri Seti Hikayesi
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
# bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
# gösterir.

# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

# Görev 1: Keşifçi Veri Analizi
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
# Adım 5: Aykırı gözlem var mı inceleyiniz.
# Adım 6: Eksik gözlem var mı inceleyiniz.


# Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# Adım 2: Yeni değişkenler oluşturunuz.
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.


# Görev 3 : Modelleme
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli
# tekrar kurunuz.


###############################
# Görev 1: Keşifçi Veri Analizi

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)
import warnings
warnings.simplefilter(action="ignore")


df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.describe().T
df.dtypes


# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    # 1- Categorical variables
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # 2- Numeric but actually categorical (class)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    # 3 - Categorical but actually each cardinal, that is, unique
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # 4 - Collect the cat_cols and num_but_cat variables
    cat_cols = cat_cols + num_but_cat

    # 5- Subtract the cardinal variable from cat_cols
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["SeniorCitizen"].max()

df.SeniorCitizen=df.SeniorCitizen.astype("object")
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df.dtypes

df.isnull().sum()

df.dropna(inplace = True)
df['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df['Churn'].replace(to_replace='No',  value=0, inplace=True)
df.head()

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
for col in cat_cols:
    print(pd.DataFrame({col: df[col].value_counts(),
                        "Ratio": 100 * df[col].value_counts() / len(df)}))
    print("######################")

df.describe().T

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def target_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"Target_Mean": dataframe.groupby(cat_col)[target].mean(),
                        "Count": dataframe[cat_col].value_counts(),
                        "Ratio": 100 * dataframe[cat_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_with_cat(df, "Churn", col)

# Adım 5: Aykırı gözlem var mı inceleyiniz.
    def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()
df.info()

# Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
df["MultipleLines"].unique()
df["InternetService"].unique()
df["OnlineSecurity"].unique()
df["OnlineBackup"].unique()
df["DeviceProtection"].unique()
df["TechSupport"].unique()
df["StreamingMovies"].unique()
df["StreamingTV"].unique()
df["PaperlessBilling"].unique()
df["PhoneService"].unique()

# Adım 2: Yeni değişkenler oluşturunuz.
# Kişi tarafından alınan toplam hizmet sayısı
df['New_Total_Services'] = (df[[ 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingMovies','StreamingTV','PhoneService']] == 'Yes').sum(axis=1)
df['New_Total_Services']

# Ortalama aylık ödeme
df["New_Avg_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["New_Avg_Charges"]

# tenure : (0-71)
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"New_Tenure_Year"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"New_Tenure_Year"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"New_Tenure_Year"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"New_Tenure_Year"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"New_Tenure_Year"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"New_Tenure_Year"] = "5-6 Year"
df["New_Tenure_Year"]

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# Label encoder
def label_encoder(dataframe, binary_cols):
    dataframe[binary_cols] = LabelEncoder().fit_transform(dataframe[binary_cols])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)

# OneHot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df2 = one_hot_encoder(df, cols)
df2.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
for col in num_cols:
    transformer = RobustScaler().fit(df2[[col]])
    df[col] = transformer.transform(df2[[col]])

df2.head()

# Görev 3 : Modelleme

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
Y = df2["Churn"]
X = df2.drop(["Churn", "customerID"], axis=1)


def base_models(X, Y):
    print("Base Models....")
    models = [('LR', LogisticRegression()),
              ("KNN", KNeighborsClassifier()),
              ("CART", DecisionTreeClassifier()),
              ("RF", RandomForestClassifier()),
              ('GBM', GradientBoostingClassifier()),
              ("XGBoost", XGBClassifier(eval_metric='logloss')),
              ("LightGBM", LGBMClassifier()),
              ("CatBoost", CatBoostClassifier(verbose=False))]

    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


def cv_results(x):
    print(f"Accuracy: {round(x['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(x['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(x['test_recall'].mean(), 4)}")
    print(f"Precision: {round(x['test_precision'].mean(), 4)}")
    print(f"F1: {round(x['test_f1'].mean(), 4)}")
base_models(X, Y)


# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli
# tekrar kurunuz.

catboost_model = CatBoostClassifier(random_state=17, verbose=False)
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
catboost_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=False).fit(X, Y)
catboost_final = catboost_model.set_params(**catboost_grid.best_params_, random_state=17).fit(X, Y)

# Final Model
cv_results = cross_validate(catboost_final, X, Y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision","recall"])

print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")