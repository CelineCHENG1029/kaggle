import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

data_train = pd.read_csv("data/train.csv")


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
plt.title("Survived") # 标题
plt.ylabel("Number")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("Number")
plt.title("Class distribution")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("Age")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title("Survived")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")# plots an axis lable
plt.ylabel("Density") 
plt.title("Age distribution")
plt.legend(('1st', '2nd','3rd'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("Embarked number")
plt.ylabel("Number")  
plt.show()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'Survived':Survived_1, 'NonSurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("Survived")
plt.xlabel("Class") 
plt.ylabel("Number") 
plt.show()

fig = plt.figure()
fig.set(alpha=0.4)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({'Male':Survived_m, 'Female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("Survived for different sexes")
plt.xlabel("Survived") 
plt.ylabel("Number")
plt.show()

fig = plt.figure()
fig.set(alpha=0.65) # 设置图像透明度
plt.title("Survived based on class and sex")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex=="female"][data_train.Pclass!=3].value_counts().plot(kind='bar',label="female highclass",color="green")
ax1.set_xticklabels(["S","N"],rotation=0)
plt.legend(["Female/HighClass"],loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, lowclass', color='pink')
ax2.set_xticklabels(["S","N"],rotation=0)
plt.legend(["Female/LowClass"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, highclass',color='lightblue')
ax3.set_xticklabels(["S","N"],rotation=0)
plt.legend(["Male/HC"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male lowclass', color='steelblue')
ax4.set_xticklabels(["S","N"],rotation=0)
plt.legend(["Male/LC"], loc='best')

plt.show()

fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'Survived':Survived_1, 'NonSurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("Survived for different ports")
plt.xlabel("Port") 
plt.ylabel("Number") 

plt.show()

g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)


g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)

data_train.Cabin.value_counts()

#fig = plt.figure()
#fig.set(alpha=0.2)
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'Survived_cabin':Survived_cabin,'Survived_nocabin':Survived_nocabin}).transpose()
df.plot(kind='bar',stacked=True)
plt.title("Survived with/out cabin")
plt.xlabel("Cabin or not")
plt.ylabel("Number")

plt.show()






