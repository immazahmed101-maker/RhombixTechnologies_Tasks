# Data handling & analysis
import pandas as pd
import numpy as np

# Visualization (EDA & evaluation)
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Titanic dataset into DataFrame 
titanic_df = pd.read_csv("Titanic-Dataset.csv")

# Previwe first 5 rows
titanic_df.head()

# Check dataset shape (rows, columns)
titanic_df.shape

# Check datatypes & missing values
titanic_df.info()

#Check missing values column-wise
titanic_df.isnull().sum()

# Drop Cabin column due to -77% missing values
titanic_df.drop('Cabin',axis=1, inplace=True)

# Fill missing Age values using median (robust to outliers)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Fill missing Embarked values using mode (most frequent part)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

#Convert Sex column to numerical format male = 0, female = 1 
titanic_df['Sex'] = titanic_df['Sex'].map({'male':0, 'female':1})

#On-hot encode Embarked column
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'], drop_first = True)

# Select input features
x = titanic_df[['Pclass','Sex','Age','Fare','SibSp','Parch']]

# Target Variable
y = titanic_df['Survived']

# Check shapes
display("x",x.shape,"y",y.shape)

#Split data into training (80%) and testing (20%)
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size = 0.2, random_state = 42)

y_pred

# Model accuracy
print('Accuracy:', accuracy_score(y_test, y_pred))

# Confussion Matrix
print(confusion_matrix(y_test, y_pred,))

# Detailed classification report
print(classification_report(y_test, y_pred))

# New passenger data (Numpy)
sample = np.array([[3,0,22,7.25,1,0]])

# Prediction survival
prediction = lg.predict(sample)

# Convert numerical output to readable text
result = np.where(prediction == 1,
                 'Passenger Survived',
                 'Passenger Did Not Survive')

print(result[0])

# Survival probability
probality = lg.predict_proba(sample)
(f'Survived Probality:{probality[0][1]:.2f}')

# Survival distribution
sns.countplot(x = 'Survived', data=titanic_df)
plt.title('Survival Distribution')
plt.show()

# Age vs Survival
sns.histplot(data=titanic_df, x = 'Age', hue ='Survived', bins=30, kde=True)
plt.title('Age Distributtion by Survival')
plt.show()

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt ='d', cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('confusion_matrix')
plt.show()
