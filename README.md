# Kaggle Titanic Machine Learning Challenge
#Machine Learning Homework
 
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200

[8 rows x 7 columns]
<class 'pandas.core.frame.DataFrame'>
Int64Index: 712 entries, 665 to 244
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  712 non-null    int64  
 1   Survived     712 non-null    int64  
 2   Pclass       712 non-null    int64  
 3   Name         712 non-null    object 
 4   Sex          712 non-null    object 
 5   Age          582 non-null    float64
 6   SibSp        712 non-null    int64  
 7   Parch        712 non-null    int64  
 8   Ticket       712 non-null    object 
 9   Fare         712 non-null    float64
 10  Cabin        164 non-null    object 
 11  Embarked     711 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 72.3+ KB
None
     PassengerId  Survived  Pclass        Age  ...    C    S    Q  Female
665          666         0       2  32.000000  ...  0.0  0.0  1.0     0.0
230          231         1       1  35.000000  ...  0.0  0.0  1.0     1.0
864          865         0       2  24.000000  ...  0.0  0.0  1.0     0.0
837          838         0       3  30.006581  ...  0.0  0.0  1.0     0.0
723          724         0       2  50.000000  ...  0.0  0.0  1.0     0.0
..           ...       ...     ...        ...  ...  ...  ...  ...     ...
70            71         0       2  32.000000  ...  0.0  0.0  1.0     0.0
672          673         0       2  70.000000  ...  0.0  0.0  1.0     0.0
473          474         1       2  23.000000  ...  1.0  0.0  0.0     1.0
128          129         1       3  30.006581  ...  1.0  0.0  0.0     1.0
244          245         0       3  30.000000  ...  1.0  0.0  0.0     0.0

[712 rows x 11 columns]


<class 'pandas.core.frame.DataFrame'>
Int64Index: 712 entries, 665 to 244
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  712 non-null    int64  
 1   Survived     712 non-null    int64  
 2   Pclass       712 non-null    int64  
 3   Age          712 non-null    float64
 4   SibSp        712 non-null    int64  
 5   Parch        712 non-null    int64  
 6   Fare         712 non-null    float64
 7   C            712 non-null    float64
 8   S            712 non-null    float64
 9   Q            712 non-null    float64
 10  Female       712 non-null    float64
dtypes: float64(6), int64(5)
memory usage: 66.8 KB
None


GridSearchCV(cv=3, estimator=RandomForestClassifier(),
             param_grid=[{'max_depth': [None, 5, 10],
                          'min_samples_split': [2, 3, 4],
                          'n_estimators': [10, 100, 200, 500]}],
             return_train_score=True, scoring='accuracy')
RandomForestClassifier(max_depth=5, min_samples_split=4, n_estimators=200)

0.8324022346368715


     PassengerId  Survived  Pclass        Age  ...    C    S    Q  Female
0              1         0       3  22.000000  ...  0.0  0.0  1.0     0.0
1              2         1       1  38.000000  ...  1.0  0.0  0.0     1.0
2              3         1       3  26.000000  ...  0.0  0.0  1.0     1.0
3              4         1       1  35.000000  ...  0.0  0.0  1.0     1.0
4              5         0       3  35.000000  ...  0.0  0.0  1.0     0.0
..           ...       ...     ...        ...  ...  ...  ...  ...     ...
886          887         0       2  27.000000  ...  0.0  0.0  1.0     0.0
887          888         1       1  19.000000  ...  0.0  0.0  1.0     1.0
888          889         0       3  29.699118  ...  0.0  0.0  1.0     1.0
889          890         1       1  26.000000  ...  1.0  0.0  0.0     0.0
890          891         0       3  32.000000  ...  0.0  1.0  0.0     0.0

[891 rows x 11 columns]

GridSearchCV(cv=3, estimator=RandomForestClassifier(),
             param_grid=[{'max_depth': [None, 5, 10],
                          'min_samples_split': [2, 3, 4],
                          'n_estimators': [10, 100, 200, 500]}],
             return_train_score=True, scoring='accuracy')
             
     PassengerId  Pclass       Age  SibSp  ...    C    S    Q  Female
0            892       3  34.50000      0  ...  0.0  1.0  0.0     0.0
1            893       3  47.00000      1  ...  0.0  0.0  1.0     1.0
2            894       2  62.00000      0  ...  0.0  1.0  0.0     0.0
3            895       3  27.00000      0  ...  0.0  0.0  1.0     0.0
4            896       3  22.00000      1  ...  0.0  0.0  1.0     1.0
..           ...     ...       ...    ...  ...  ...  ...  ...     ...
413         1305       3  30.27259      0  ...  0.0  0.0  1.0     0.0
414         1306       1  39.00000      0  ...  1.0  0.0  0.0     1.0
415         1307       3  38.50000      0  ...  0.0  0.0  1.0     0.0
416         1308       3  30.27259      0  ...  0.0  0.0  1.0     0.0
417         1309       3  30.27259      1  ...  1.0  0.0  0.0     0.0

[418 rows x 10 columns]
     PassengerId  Survived
0            892         0
1            893         0
2            894         0
3            895         0
4            896         1
..           ...       ...
413         1305         0
414         1306         1
415         1307         0
416         1308         0
417         1309         1

[418 rows x 2 columns]

![readme1figure](https://user-images.githubusercontent.com/98085177/229853109-17dc5b43-50ea-40cf-9f82-da683326d6e7.png)
![readme2figure](https://user-images.githubusercontent.com/98085177/229853343-f9008453-7c62-4bae-8b89-e9bc98a3114a.png)


