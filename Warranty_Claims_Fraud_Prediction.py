# -*- coding: utf-8 -*-
# Auto-converted from Jupyter Notebook (.ipynb)
# Source: Warranty Claims Fraud Prediction.ipynb

# %% [markdown] (cell 1)
# # Warranty Claims Fraud Prediction

# %% [markdown] (cell 2)
# The aim of this project is to analyze the warranty claims based on their region, product, claim value and other features to predict their authenticity. The dataset is taken from Kaggle. The dataset contains 358 rows and 21 columns. 
# 
# ### Data Dictionary
# | Column Name | Description |
# | --- | --- |
# |Unnamed: 0| Index|
# |Region| Region of the claim|
# |State| State of the claim|
# |Area| Area of the claim|
# |City| City of the claim|
# |Consumer_profile| Consumer profile Business/Personal|
# |Product_category| Product category Household/Entertainment|
# |Product_type| Product type AC/TV|
# |AC_1001_Issue| 1 0- No issue / No componenent, 1- repair, 2-replacement|
# |AC_1002_Issue| 1 0- No issue / No componenent, 1- repair, 2-replacement|
# |AC_1003_Issue| 1 0- No issue / No componenent, 1- repair, 2-replacement|
# |TV_2001_Issue| 1 0- No issue / No componenent, 1- repair, 2-replacement|
# |TV_2002_Issue| 1 0- No issue / No componenent, 1- repair, 2-replacement|
# |TV_2003_Issue| 1 0- No issue / No componenent, 1- repair, 2-replacement|
# |Claim_Value| Claim value in INR|
# |Service_Center| Service center code|
# |Product_Age| Product age in days|
# |Purchased_from| Purchased from - Dealer, Manufacturer, Internet|
# |Call_details| Call duration|
# |Purpose| Purpose of the call|
# |Fraud| Fraudulent (1) or Genuine (0)|

# %% (cell 3)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% (cell 4)
#Loading the dataset
df = pd.read_csv('df_Clean.csv')
df.head()

# %% [markdown] (cell 5)
# ## Data Preprocessing Part 1

# %% (cell 6)
# checking the shape of the dataset
df.shape

# %% (cell 7)
# Drop index column
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# %% (cell 8)
# Checking for null/missing values
df.isnull().sum()

# %% (cell 9)
# Checking for duplicate values
df.duplicated().sum()

# %% (cell 10)
# Checking the data types
df.dtypes

# %% (cell 11)
# Unique values in each column
df.nunique()

# %% (cell 12)
# renaming the values in product issue column
df['AC_1001_Issue'] = df['AC_1001_Issue'].map({ 0 : 'No Issue', 1 : 'repair', 2 : 'replacement'})
df['AC_1002_Issue'] = df['AC_1002_Issue'].map({ 0 : 'No Issue', 1 : 'repair', 2 : 'replacement'})
df['AC_1003_Issue'] = df['AC_1003_Issue'].map({ 0 : 'No Issue', 1 : 'repair', 2 : 'replacement'})
df['TV_2001_Issue'] = df['TV_2001_Issue'].map({ 0 : 'No Issue', 1 : 'repair', 2 : 'replacement'})
df['TV_2002_Issue'] = df['TV_2002_Issue'].map({ 0 : 'No Issue', 1 : 'repair', 2 : 'replacement'})
df['TV_2003_Issue'] = df['TV_2003_Issue'].map({ 0 : 'No Issue', 1 : 'repair', 2 : 'replacement'})

# %% [markdown] (cell 13)
# #### Descriptive Statistics

# %% (cell 14)
df.describe()

# %% (cell 15)
df.head()

# %% [markdown] (cell 16)
# ## Exploratory Data Analysis

# %% [markdown] (cell 17)
# ### Location based Distribution of Fraudulent Claims

# %% (cell 18)
fig, ax = plt.subplots(2,2,figsize=(15,10))
fig.subplots_adjust(hspace=0.7)

sns.histplot(x = 'Region', data = df, ax =ax[0,0], hue = 'Fraud', element='bars', fill=True, stat='density',multiple='stack').set(title='Regional Distribution of Fraudulent Claims')
ax[0,0].xaxis.set_tick_params(rotation=90)

sns.histplot(x = 'State', data = df, ax =ax[0,1], hue = 'Fraud', element='bars', fill=True, stat='density',multiple='stack').set(title='Statewise Distribution of Fraudulent Claims')
ax[0,1].xaxis.set_tick_params(rotation=90)

sns.histplot(x = 'City', data = df, ax =ax[1,0], hue = 'Fraud', element='bars', fill=True, stat='density',multiple='stack').set(title='Citywise Distribution of Fraudulent Claims')
ax[1,0].xaxis.set_tick_params(rotation=90)

sns.histplot(x = 'Area', data = df, ax =ax[1,1], hue = 'Fraud', element='bars', fill=True, stat='density',multiple='stack').set(title='Areawise Distribution of Fraudulent Claims')

# %% [markdown] (cell 19)
# The above plots visualizes the distribution of fraudulent claims based on location. The first graphs shpws the regional distribution of the fraudent claims, where South, North East and south East are among the regionas with highest warranty claims. However, the regions - West, East and South are among regions with highest fraudulent claims. Interestingly the North West region has zero fraudent claims. 
# 
# The second graph shows the distribution of fraudulent claims based on the States, where the states - Andhra Pradesh, Maharashtra, Tamil Nadu, Karnataka and Gujarat are among the states with highest number of warranty claims and states - Haryana has lowest warranty claims. The states - Andhra Pradesh, Maharashtra, Tamil Nadu, Karnataka and Gujarat are among the states with highest number of fraudulent claims whereas, states like Bihar, Delhi, West Bengal and Assam are among the states with lowest number of fraudulent claims.
# 
# The third graph shows the distribution of fraudulent claims based on cities. The cities - Chennai, Hyderabad, Bangalore, Mumbai and Kochi are among the cities with highest claims whereas cities like Chandigarh, Srinagar, Agartala and Shimla have lowest number of claims. Moreover the cities - Chennai, Hyderabad, Bangalore, Mumbai and Kochi are among the cities with highest fraudulent claims whereas cities like Chandigarh, Panaji, Meerut, Jaipur, and many other have zero fraudulent claims.
# 
# The forth graph, visualizes the fraudulent claims based on the area, where the urban area has more number of claims and ultimately more number of fraudulent claims in comparison to rural areas.

# %% [markdown] (cell 20)
# ### Consumer Profile and Fraudulent Claims

# %% (cell 21)
sns.countplot(x = 'Consumer_profile', data = df, hue = 'Fraud').set_title('Consumer Profile distribution')

# %% [markdown] (cell 22)
# From this graph, it is clear that majority of the claims are from consumer who purchased the products for personal use. However,the consumers who purchases the products for business purpose have higher number of fraudulent warranty claims.

# %% [markdown] (cell 23)
# ### Product and Fraudulent Claims

# %% (cell 24)
sns.histplot(x = 'Product_type', data = df, hue = 'Fraud', multiple='stack').set_title('Product and Fraud Distribution')

# %% [markdown] (cell 25)
# This graph shows that the company has higher sales for the TV as compared to the AC, and ultimately the number warranty claims for TV is higher than AC. Moreover, the number of fraudulent claims for TV is also higher than AC.

# %% [markdown] (cell 26)
# ### Issue with the Product Parts and Fraudulent Claims

# %% (cell 27)
fig, ax = plt.subplots(2,3,figsize=(20,12))
sns.histplot(x = 'AC_1001_Issue', data = df, ax =ax[0,0], hue = 'Fraud', multiple='stack').set(title='AC_1001_Issue and Fraud Distribution')

sns.histplot(x = 'AC_1002_Issue', data = df, ax =ax[0,1], hue = 'Fraud', multiple='stack').set(title='AC_1002_Issue and Fraud Distribution')

sns.histplot(x = 'AC_1003_Issue', data = df, ax =ax[0,2], hue = 'Fraud', multiple='stack').set(title='AC_1003_Issue and Fraud Distribution')

sns.histplot(x = 'TV_2001_Issue', data = df, ax =ax[1,0], hue = 'Fraud', multiple='stack').set(title='TV_2001_Issue and Fraud Distribution')

sns.histplot(x = 'TV_2002_Issue', data = df, ax =ax[1,1], hue = 'Fraud', multiple='stack').set(title='TV_2002_Issue and Fraud Distribution')

sns.histplot(x = 'TV_2003_Issue', data = df, ax =ax[1,2], hue = 'Fraud', multiple='stack').set(title='TV_2003_Issue and Fraud Distribution')

# %% [markdown] (cell 28)
# The above graphs visualizes the issue with the product parts and fradulent warranty claims on them. In the product AC the parts AC_1001 and AC_1002 has increases number of repairs whereas as the AC_1003 has considerable less instances of repair or replacement as compared to other two, so the company should focus on improving the AC_1001 and AC_1002 parts. Moreover, in all three parts, fradulent claims usually occurs when there is no issue with the product.
# 
# In the product TV the parts TV_2001 and TV_2002 has increases number of repairs whereas as the TV_1003 has considerable less instances of repair and negligible instances of replacement as compared to other two, however in contrast to AC, the fradulent claims usually occurs when there is issue with the product as well as when the product parts especially TV_2001 and TV_2002 requires repair or replacement. So the company should focus on improving the TV_2001 and TV_2002 parts, in order to reduce the number of fradulent claims.

# %% [markdown] (cell 29)
# ### Service Center and Fraudulent Claims

# %% (cell 30)
sns.countplot(x = 'Service_Centre', data = df, hue = 'Fraud').set_title('Service Centre and Fraudulent Claims')

# %% [markdown] (cell 31)
# This graoh shows the relation between the relationship between the service centre and the fraudulent warranty claims. The majorty of the replairs and replacements are done by the service centre 15,12 and 13. Where, the service centre 13 has the highest number of fradulent claims, followed by service centre 10. So, the company should survelliance the service centre 13 and 10 more closely.

# %% [markdown] (cell 32)
# ### Claim Value and Fraudulent Claims

# %% (cell 33)
fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x = 'Fraud', y = 'Claim_Value', data = df, ax =ax[0]).set_title('Claim Value and Fraudulent Claims')

sns.violinplot(x = 'Fraud', y = 'Claim_Value', data = df, ax =ax[1]).set_title('Claim Value and Fraudulent Claims')

# %% [markdown] (cell 34)
# As expected, these graphs shows that the claim value for fradulent clains tends to be higher than the genuine claims. In the boxplot, the medianclaim value of fraudulent claims is way higher than the genuine claims. In addtion to that, it is clear form the boxplot that the fraudulent claims are more spread out at higher claim values than the genuine claims.

# %% [markdown] (cell 35)
# ### Product Age and Fraudulent Claims

# %% (cell 36)
sns.histplot(x = 'Product_Age', data = df, hue = 'Fraud', multiple='stack', bins = 20).set_title('Product Age(in days) and Fraud Distribution')

# %% [markdown] (cell 37)
# From the above histogram, it is clear that majority of the warranty claims occur within 100 days of purchase. However, the fraudulent claims are more frequent and they usually occur within 50 days of purchase.

# %% [markdown] (cell 38)
# ### Purchase point and Fraudulent Claims

# %% (cell 39)
sns.histplot(x = 'Purchased_from', data = df, hue = 'Fraud', multiple='stack').set_title('Purchased from and Fraudulent Claims')

# %% [markdown] (cell 40)
# Maximum number of purchase is done through the dealer, but the maximum number of fraudulent claims are coming when the purchase is done through the manufacturer, whereas the internet has the lowest number of fraudulent claims. This much fradulent claims only from the manufacturer is a matter of concern for the company.

# %% [markdown] (cell 41)
# ### Call Duration and Fraudulent Claims

# %% (cell 42)
sns.histplot(x = 'Call_details', data = df, hue = 'Fraud', multiple='stack').set_title('Call Duration and Fraudulent Claims')
plt.xlabel('Call Duration(in mins)')

# %% [markdown] (cell 43)
# This graph shows the relation of customer care call duration and the fraudulent claims. In order to make a warranty claims, customers contact the customer care. The duration of customer care calls are plotted in the histogram along with the authenciity of the claims. The histogram shows that the fraudulent claims are more frequent when the customer care call duration is less than 3-4 minutes. However, the genuine claims are more frequent when the customer care call duration is more than 4 minutes.

# %% [markdown] (cell 44)
# ### Purpose of contact and Fraudulent Claims

# %% (cell 45)
sns.histplot(x = 'Purpose', data = df, hue = 'Fraud', multiple='stack').set_title('Purpose and Fraudulent Claims')

# %% [markdown] (cell 46)
# Most of the customer contact the customer care for the purpose of complaint and claim and very few with other reasons. However, the fraudulent claims are more frequent when the customer contact the customer care for the purpose of complaint and claim.

# %% [markdown] (cell 47)
# ## Data Preprocessing Part 2

# %% [markdown] (cell 48)
# ### Outlier Removal

# %% (cell 49)
# Removing outliners from claim value column using IQR method

Q1 = df['Claim_Value'].quantile(0.25)
Q3 = df['Claim_Value'].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df['Claim_Value'] < (Q1 - 1.5 * IQR)) |(df['Claim_Value'] > (Q3 + 1.5 * IQR)))]

# %% [markdown] (cell 50)
# ### Label Encoding the Object Datatypes

# %% (cell 51)
from sklearn.preprocessing import LabelEncoder

#Label encoding Object
le = LabelEncoder()

# columns for label encoding
cols = df.select_dtypes(include=['object']).columns

# label encoding
for col in cols:
    le.fit(df[col])
    df[col] = le.transform(df[col])
    print(col, df[col].unique())

# %% [markdown] (cell 52)
# ## Correlation Matrix Heatmap

# %% (cell 53)
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# %% [markdown] (cell 54)
# ## Train Test Split

# %% (cell 55)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Fraud',axis=1), df['Fraud'], test_size=0.30, random_state=42)

# %% [markdown] (cell 56)
# ## Model Building
# 
# I will be using the following classification models:
# - Decision Tree Classifier
# - Random Forest Classifier
# - Logistic Regression

# %% [markdown] (cell 57)
# ### Decision Tree Classifier

# %% (cell 58)
from sklearn.tree import DecisionTreeClassifier

#Decision Tree Classifier Object
dtree = DecisionTreeClassifier()

# %% [markdown] (cell 59)
# Hyperparameter Tuning using GridSearchCV

# %% (cell 60)
from sklearn.model_selection import GridSearchCV

#parameters for grid search
param_grid = {
    'max_depth': [2,4,6,8,10],
    'min_samples_leaf': [2,4,6,8,10],
    'min_samples_split': [2,4,6,8,10],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
}

#Grid Search Object with Decision Tree Classifier
grid = GridSearchCV(dtree, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')

#Fitting the grid search object to the training data
grid.fit(X_train,y_train)

#Best parameters for Decision Tree Classifier
print(grid.best_params_)

# %% (cell 61)
#Best estimator for Decision Tree Classifier
dtree = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=2, min_samples_split=2, random_state=0)

#Fitting the Decision Tree Classifier to the training data
dtree.fit(X_train,y_train)

#training accuracy
print(dtree.score(X_train,y_train))

#prediction on test data
d_pred = dtree.predict(X_test)

# %% [markdown] (cell 62)
# ### Random Forest Classifier

# %% (cell 63)
from sklearn.ensemble import RandomForestClassifier

#Random Forest Classifier Object
rfc = RandomForestClassifier()

# %% [markdown] (cell 64)
# Hyperparameter Tuning using GridSearchCV

# %% (cell 65)
from sklearn.model_selection import GridSearchCV

#parameters for grid search
param_grid = {
    'max_depth': [2,4,6,8],
    'min_samples_leaf': [2,4,6,8],
    'min_samples_split': [2,4,6,8],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
}

#Grid Search Object with Random Forest Classifier
grid = GridSearchCV(rfc, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')

#Fitting the grid search object to the training data
grid.fit(X_train,y_train)

#Best parameters for Random Forest Classifier
print(grid.best_params_)

# %% (cell 66)
#random forest classifier with best parameters
rfc = RandomForestClassifier(criterion='gini', max_depth=2, min_samples_leaf=2, min_samples_split=2, random_state=0)

#Fitting the Random Forest Classifier to the training data
rfc.fit(X_train,y_train)

#training accuracy
print(rfc.score(X_train,y_train))

#prediction on test data
r_pred = rfc.predict(X_test)

# %% [markdown] (cell 67)
# ### Logistic Regression

# %% (cell 68)
from sklearn.linear_model import LogisticRegression

#Logistic Regression Object
lr = LogisticRegression()

#Fitting the Logistic Regression to the training data
lr.fit(X_train,y_train)

#training accuracy
print(lr.score(X_train,y_train))

#prediction on test data
l_pred = lr.predict(X_test)

# %% [markdown] (cell 69)
# ## Model Evaluation

# %% [markdown] (cell 70)
# ### Confusion Matrix Heatmap

# %% (cell 71)
fig, ax = plt.subplots(1,3,figsize=(20,5))

from sklearn.metrics import confusion_matrix

#confusion matrix for Decision Tree Classifier
sns.heatmap(confusion_matrix(y_test,d_pred), annot=True, cmap='coolwarm', ax=ax[0]).set_title('Decision Tree Classifier')

#confusion matrix for Random Forest Classifier
sns.heatmap(confusion_matrix(y_test,r_pred), annot=True, cmap='coolwarm', ax=ax[1]).set_title('Random Forest Classifier')

#confusion matrix for Logistic Regression
sns.heatmap(confusion_matrix(y_test,l_pred), annot=True, cmap='coolwarm', ax=ax[2]).set_title('Logistic Regression')

# %% [markdown] (cell 72)
# ### Classification Report

# %% (cell 73)
from sklearn.metrics import classification_report

#classification report for Decision Tree Classifier
print(classification_report(y_test,d_pred))

#classification report for Random Forest Classifier
print(classification_report(y_test,r_pred))

#classification report for Logistic Regression
print(classification_report(y_test,l_pred))

# %% (cell 74)
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

print('==================== Decision Tree Classifier ====================')
print('Accuracy Score: ', accuracy_score(y_test,d_pred))
print('R2 Score: ', r2_score(y_test,d_pred))
print('Mean Squared Error: ', mean_squared_error(y_test,d_pred))

print('==================== Random Forest Classifier ====================')
print('Accuracy Score: ', accuracy_score(y_test,r_pred))
print('R2 Score: ', r2_score(y_test,r_pred))
print('Mean Squared Error: ', mean_squared_error(y_test,r_pred))

print('==================== Logistic Regression =========================')
print('Accuracy Score: ', accuracy_score(y_test,l_pred))
print('R2 Score: ', r2_score(y_test,l_pred))
print('Mean Squared Error: ', mean_squared_error(y_test,l_pred))

# %% [markdown] (cell 75)
# ## Feature Importance

# %% (cell 76)
#feature importance for Decision Tree Classifier
feature_importance_d = pd.DataFrame(dtree.feature_importances_, index=X_train.columns, columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)

#feature importance for Random Forest Classifier
feature_importance_r = pd.DataFrame(rfc.feature_importances_, index=X_train.columns, columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)

fig, ax = plt.subplots(1,2,figsize=(20,5))
#space between subplots
fig.subplots_adjust(wspace=0.5)
sns.barplot(y=feature_importance_d.index, x=feature_importance_d['Feature Importance'], ax=ax[0]).set_title('Decision Tree Classifier')
ax[0].xaxis.set_tick_params(rotation=90)
sns.barplot(y=feature_importance_r.index, x=feature_importance_r['Feature Importance'], ax=ax[1]).set_title('Random Forest Classifier')
ax[1].xaxis.set_tick_params(rotation=90)

# %% [markdown] (cell 77)
# ## Conclusion
# 
# From the exploratory data analysis, I have concluded that most of the warranty claims takes place in the southern region of India particularly in Andhra Pradesh and Tamil Nadu. Moreover, the fraudulent claims are more frequent in the cities like Hyderabad and Chennai whih are urban regions. The dataset includes the claims regarding two products i.e. TV and AC. The TVs had the higher warranty claims when they where purchased for personal purposes as compared to AC.
# 
# Moreover, in the case of Ac the fraudulent claims were made, when there was no issue in the AC parts. However, in the case of TV the fraudulent claims were made, when there was issue in the TV parts as well as when there was no issue in the TV parts. The fraudulent claims were more frequent when the purchase was made through the manufacturer. 
# 
# The fraudulent claims tend to have higher claim value as compared to the genuine ones, and the service centre 13 had the highest number of fraudulent claims despite of having lesser number of total warranty claims. It was also observed that the fraudulent claims were more frequent when the customer care call duration was less than 3-4 minutes. 
# 
# 
# Coming to the machine learning models, I have used Decision Tree Classifier, Random Forest Classifier and Logistic Regression. All these models gave excellent accuracy of 91-92%. However, due to lesser number of fraudulent claims or small dataset size, the models have poor recall score for fraudulent claims. But this issue can be resolved by collecting more data.
