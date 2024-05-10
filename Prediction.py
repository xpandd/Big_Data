#!/usr/bin/env python
# coding: utf-8

# # <font color='#800080'>NEURONINIAI TINKLAI</font>               
# ### <font>*Asta Gražytė-Skominienė*</font>
# ________________________________________________________________________________________________________________________________

#pip install lightgbm
#pip install catboost --no-cache-dir
#pip install tensorflow
#pip install scikeras[tensorflow]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
#from tensorflow import keras
#from scikeras.wrappers import KerasRegressor
#import tensorflow as tf


df = pd.read_csv("anonymized_full_release_competition_dataset.csv", sep=',', low_memory=False)

df.info()

df.drop(df.columns[[31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]], axis=1, inplace=True)
df.drop(df.columns[[14, 3, 31, 32, 33, 34, 35, 36, 37, 51, 52, 53, 54, 55, 56]], axis=1, inplace=True)
df.drop(df.columns[[6,14,16,17,42,43]], axis=1, inplace=True)

#data1 = df.copy()
df= df.dropna()

print(df.describe())

# Patikrinti hipotezę: Ar 'AveCorrect' ir 'AveResBored' yra statistiškai reikšmingai skirtingi
t_statistic, p_value = stats.ttest_rel(df['AveCorrect'], df['AveResBored'])

print(f"Porinis t-testo statistika: {t_statistic}, p-reikšmė: {p_value}")

if p_value < 0.05:
    print("Yra statistinių įrodymų, kad pasiekimai ir nuobodulio lygis yra statistiškai reikšmingai skirtingi.")
else:
    print("Nėra statistinių įrodymų, kad pasiekimai ir nuobodulio lygis būtų statistiškai reikšmingai skirtingi.")



# Emocinės būsenos stulpeliai
emotional_states = ['AveResEngcon', 'AveResConf', 'AveResFrust', 'AveResGaming', 'AveResBored', 'AveCarelessness', 'AveResOfftask', "AveCorrect", "AveKnow"]

# Atlikti porinį t-testą tarp 'AveCorrect' ir kiekvienos emocinės būsenos
results = []
for state in emotional_states:
    t_statistic, p_value = stats.ttest_rel(df['AveCorrect'], df[state])
    results.append({
        'Emotional State': state,
        'T-Statistic': t_statistic,
        'P-Value': p_value
    })

# Rezultatų išvedimas
results_df = pd.DataFrame(results)
print(results_df)


# Rezultatai parodo, kad kiekviena emocinė būsena turi statistiškai reikšmingą skirtumą palyginus su mokinių pasiekimais (AveCorrect), nes visų emocinių būsenų 
# 𝑝
# p-reikšmės yra žemesnės už 0.05, faktiškai rodydamos nulinę 
# 𝑝
# p-reikšmę dėl didelio imties dydžio.
# 
# Interpretacija:
# Neigiami T-Statistikos rezultatai: Emocinės būsenos, tokios kaip AveResEngcon (koncentracija), rodantys neigiamus T-statistikos rezultatus, reiškia, kad didesnis šios būsenos rodiklis yra susijęs su mažesniais mokinių pasiekimais, arba, kad didesni mokinių pasiekimai yra susiję su mažesniais šios emocinės būsenos rodikliais.
# Teigiami T-Statistikos rezultatai: Būsenos, tokios kaip AveResConf (sumišimas) ir AveCarelessness (neatsargumas), su teigiamais rezultatais rodo, kad didesni šių rodiklių reikšmės yra susijusios su didesniais mokinių pasiekimais, arba, kad mažesni šių rodiklių lygiai yra susiję su mažesniais pasiekimais.
# Ką tai reiškia praktikoje:
# Šie rezultatai gali rodyti, kad kai kurios emocinės būsenos turi aiškią sąveiką su mokymosi sėkme ar nesėkme. Pavyzdžiui:
# 
# AveResEngcon (Koncentracija): Mažesni koncentracijos lygiai gali būti susiję su geresniais mokinių pasiekimais, o tai gali būti netikėta. Tai galėtų reikalauti papildomos analizės ar interpretacijos, nes intuitiškai tikėtumėmės priešingos tendencijos.
# AveResConf (Sumišimas): Didesni sumišimo lygiai susiję su geresniais mokinių pasiekimais, kas taip pat gali atrodyti prieštaringa. Tai gali reikšti, kad mokiniai, kurie yra sumišę, galbūt daugiau stengiasi arba daugiau dėmesio skiria mokymuisi, nors ir susiduria su sunkumais.
# Šie rezultatai turėtų būti vertinami atsargiai, nes gali būti, kad yra kitų kintamųjų, kurie įtakoja šiuos ryšius, arba kad mokinių atsakymų matavimo būdas gali neteisingai atspindėti tikrąjį emocinės būsenos poveikį. Tokiu atveju rekomenduojama atlikti papildomus tyrimus, kad būtų patikslinta, kaip šie emociniai rodikliai iš tikrųjų veikia mokinių mokymosi rezultatus.



# Duomenų paruošimas
X = df[['AveResEngcon', 'AveResConf', 'AveResFrust', 'AveResGaming', 'AveResBored', 'AveCarelessness', 'AveResOfftask']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA su 3 komponentais
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Sujunkite PCA rezultatus su kitu svarbiu stulpeliu
final_df = pd.concat([principal_df, df[['AveCorrect']]], axis=1)

# 3D vizualizacija
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(final_df['PC1'], final_df['PC2'], final_df['PC3'], c=final_df['AveCorrect'], cmap='bone', s=50, edgecolors='k')

# Rodyklių piešimas
scale_factor = 0.8  # Čia galite keisti rodyklių dydžio mažinimo koeficientą
components = pca.components_.T * np.max(principal_components) * scale_factor / np.max(pca.components_)
for i in range(components.shape[0]):
    ax.quiver(0, 0, 0, components[i, 0], components[i, 1], components[i, 2], color='red', linewidth=2, alpha=0.5)
    ax.text(components[i, 0], components[i, 1], components[i, 2], X.columns[i], color='black', ha='center', va='center', fontsize=9)

# Pakeisti fono spalvą į baltą
ax.set_facecolor('white')  # Nustato plot'o foną į baltą
fig.patch.set_facecolor('white')  # Nustato viso figūros fono spalvą į baltą

# Grafiko elementų nustatymai
ax.set_xlabel('PC 1 (51,54 %)', fontsize=10)
ax.set_ylabel('PC 2 (16,12 %)', fontsize=10)
ax.set_zlabel('PC 3 (14,44 %)', fontsize=10, rotation=280)
plt.title('PCA rezultatai', fontsize=12)
cbar = plt.colorbar(scatter, label='AveCorrect')
cbar.solids.set_edgecolor("face")
plt.grid(True)
ax.view_init(elev=20., azim=210)

plt.savefig('pca_plot_white_background2.png', format='png', dpi=300)
plt.show()


# Pvz., Koreliacinė analizė
correlation_coefficient, p_value = stats.pearsonr(df['AveResEngcon'], df['AveKnow'])

# Interpretuoti rezultatus
print("Koreliacijos koeficientas:", correlation_coefficient)
print("p reikšmė:", p_value)

if p_value < 0.05:
    print("Yra statistiškai reikšmingas ryšys tarp įsitraukimo lygio ir mokymosi rezultatų.")
else:
    print("Nėra statistiškai reikšmingo ryšio tarp įsitraukimo lygio ir mokymosi rezultatų.")



# Nustatykime priklausomą kintamąjį (mokymosi rezultatus) ir nepriklausomus kintamuosius (įsitraukimo lygį ir emocijas)
X = df[['AveResEngcon', 'AveResConf', 'AveResFrust', 'AveResGaming', 'AveResBored', 'AveCarelessness', 'AveResOfftask']]
y = df['AveCorrect']

# Pridekime konstantą į nepriklausomų kintamųjų rinkinį
X = sm.add_constant(X)

# Sukurkime daugiakintamųjų regresijos modelį
model = sm.OLS(y, X).fit()

# Gauti regresijos koeficientus ir statistinius rodiklius
print(model.summary())


# Išskirti funkcines ir nelinines funkcines funkcijas
X = df[['AveResEngcon', 'AveResConf', 'AveResFrust', 'AveResGaming', 'AveResBored', 'AveCarelessness', 'AveResOfftask']]

# Standartizuokite duomenis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA analizė
pca = PCA()  # Nustatykite, kiek pagrindinių komponentų norite išgauti
principal_components = pca.fit_transform(X_scaled)

# Paaiškinamasis kiekis kiekvienai pagrindinei komponentei
explained_variance_ratio = pca.explained_variance_ratio_

# Spausdiname kiekvieno pagrindinio komponento paaiškinamąją kiekį
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Pagrindinio komponento {i+1} paaiškinamasis kiekis: {ratio:.2%}")


data1 = df.copy()
df= df.dropna()


# ## <font color='#800080'>Duomenų paruošimas modeliui</font>


cats = [c for c in data1.columns if data1[c].dtypes=='object']
cats


# ### <font color='#800080'>Kategoriniams duomenims taikomas One-Hot Encoding</font>

del data1['studentId']
del data1['problemType']
del data1['skill']

data = pd.get_dummies(data1)

data1 = data.iloc[:len(data), ]

column_index = data.columns.get_loc('AveKnow')
print(column_index)


# Multiple Linear Regression 



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Pasirenkami kintamieji
X = data[["AveCarelessness", "AveResBored", "AveResEngcon", "AveResConf", "AveResFrust", "AveResOfftask", "AveResGaming", "correct", "hint", "hintCount", "hintTotal", "attemptCount", "manywrong", "confidence(CONFUSED)", "confidence(GAMING)", "RES_CONCENTRATING", "RES_CONFUSED"]] 
y = data['AveKnow']

# Duomenų skaidymas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standartizavimas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelio sukūrimas ir treniravimas
model = LinearRegression()
model.fit(X_train, y_train)

# Koeficientų ir sankirtos atspausdinimas
print('Koeficientai:', model.coef_)
print('Intercept:', model.intercept_)

# Modelio vertinimas su testavimo duomenimis
y_pred = model.predict(X_test)
print('Testavimo R-kvadrato reikšmė:', r2_score(y_test, y_pred))
print('Testavimo RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# Multikolinearumo tikrinimas
X_train_with_constant = sm.add_constant(X_train)  # Pridedama konstanta
X_train_with_constant = pd.DataFrame(X_train_with_constant, columns=['const'] + X_train.columns.tolist())  # Svarbu išlaikyti DataFrame formatą

vif = pd.DataFrame()
vif["features"] = X_train_with_constant.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train_with_constant.values, i) for i in range(X_train_with_constant.shape[1])]
print(vif.round(2))

# Modelio statistika naudojant statsmodels
ols_model = sm.OLS(y_train, X_train_with_constant)
results = ols_model.fit()
print(results.summary())


#Boosting regression



from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Pasirenkami kintamieji
X = data.drop('AveKnow', axis=1)  # Pakeisti 'AveKnow' į tikslinio kintamojo stulpelio pavadinimą
y = data['AveKnow']  # Pakeisti 'AveKnow' į tikslinio kintamojo stulpelio pavadinimą

# Duomenų skaidymas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost modelio sukūrimas
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hiperparametrų derinimo parametrų nustatymas
param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 500, 1000]
}

# Naudokite RandomizedSearchCV vietoje GridSearchCV
random_search = RandomizedSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_iter=10, random_state=42)
random_search.fit(X_train, y_train)

# Geriausi parametrai ir modelio vertinimas
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Geriausi hiperparametrai:", random_search.best_params_)
print("Testavimo MSE:", mse)
print("Testavimo R-kvadratas:", r2)

# Kryžminė patikra
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Kryžminės patikros MSE vidurkis:", -cv_scores.mean())

# Feature Importance Visualization
xgb.plot_importance(best_model)
plt.title("Feature Importance")
plt.show()

# Actual vs Predicted Values Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Diagonal line for reference
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='red', lw=4)  # Line at 0 for reference
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()


# Feature Importance Visualization
fig, ax = plt.subplots(figsize=(11, 6))  # Padidinkite grafiko dydį
xgb.plot_importance(best_model, ax=ax, height=0.5, color='blue', alpha=0.5)
plt.title("Feature Importance")
plt.savefig('xbgboost_feature.jpg')
plt.show()


# Actual vs Predicted Values Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, s=200)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', lw=1, linestyle='dotted')  # Diagonal line for reference
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.savefig('xgboost_pred.jpg')
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='blue', lw=1, linestyle='dotted')  # Line at 0 for reference
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.savefig('xboost_resid.jpg')
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Įvertinkite modelio metrikas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

# Duomenų Paskirstymo Analizė
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(y_test, bins=30, kde=True, color='blue', label='Actual')
plt.title('Actual Values Distribution')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(y_pred, bins=30, kde=True, color='orange', label='Predicted')
plt.title('Predicted Values Distribution')
plt.legend()
plt.savefig('xboost_count.jpg')
plt.show()


# Modelio Kompleksiškumo Mažinimas
# Čia nurodomas pavyzdinis kodas, kur galite sumažinti modelio mokymosi greitį ir padidinti reguliarizaciją
best_model.set_params(learning_rate=0.01, reg_alpha=0.1, reg_lambda=0.1)
best_model.fit(X_train, y_train)


