#importing dependencies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

#importing file...check whether the dataset file is in the same directory or change the directory to the file 
df = pd.read_csv("C:/Users/admin/Downloads/customer_dataset.csv")

#info about our dataset 
df.info()
#dropping unneccessary data
df.drop(["CustomerID"],axis = 1, inplace = True)

#1.bar plot for age
sns.axes_style("whitegrid")
plt.title("Age frequency")
sns.distplot(x=df["Age"],bins = 15)
plt.show()

#2.male vs female
gender = df.Gender.value_counts()
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.figure(figsize = (6,6))
sns.barplot(x=gender.index, y=gender.values)
plt.show()

#3.age vs customer
age1 = df.Age[(df.Age >= 15) & (df.Age <= 20)]
age2 = df.Age[(df.Age >= 21) & (df.Age <= 30)]
age3 = df.Age[(df.Age >= 31) & (df.Age <= 40)]
age4 = df.Age[(df.Age >= 41) & (df.Age <= 50)]
age5 = df.Age[df.Age >= 51]
x = ["15-20","21-30","31-40","41-50","51+"]
y = [len(age1),len(age2),len(age3),len(age4),len(age5)]
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
sns.barplot(x=x,y=y,palette = "pastel")
plt.title("Age VS Customer")
plt.xlabel("Age")
plt.ylabel("No. of Customers")
plt.show()

#4.spending score 
ss1 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >=1) & (df["Spending Score (1-100)"] <=20)]
ss2 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >=21) & (df["Spending Score (1-100)"] <=40)]
ss3 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >=41) & (df["Spending Score (1-100)"] <=60)]
ss4 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >=61) & (df["Spending Score (1-100)"] <=80)]
ss5 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >=81) & (df["Spending Score (1-100)"] <=100)]
x = ["1-20","21-40","41-60","61-80","81-100"]
y = [len(ss1),len(ss2),len(ss3),len(ss4),len(ss5)]
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
sns.barplot(x=x,y=y,palette = "mako")
plt.title("Spending Score (1-100)")
plt.xlabel("Score")
plt.ylabel("No. of Customers")
plt.show()

#5.Anuual score 
ai1 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >=1) & (df["Annual Income (k$)"] <=20)]
ai2 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >=21) & (df["Annual Income (k$)"] <=40)]
ai3 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >=41) & (df["Annual Income (k$)"] <=60)]
ai4 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >=61) & (df["Annual Income (k$)"] <=80)]
ai5 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >=81) & (df["Annual Income (k$)"] <=100)]
ai6 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >101) & (df["Annual Income (k$)"] <=120)]
ai7 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >=121) & (df["Annual Income (k$)"] <=150)]
x = ["$ 0-30k","$ 31k-40k","$ 41k-60k","$ 61k-80k","$ 81k-100k","$ 101k-120k","$ 121k-150k"]
y = [len(ai1),len(ai2),len(ai3),len(ai4),len(ai5),len(ai6),len(ai7)]
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
sns.barplot(x=x,y=y,palette = "viridis")
plt.title("Annual income (k$)")
plt.xlabel("Annual Income")
plt.ylabel("No. of Customers")
plt.show()

#6.spending score vs annual score
fig, ax = plt.subplots(figsize=(15, 6))
plt.subplot(1,2,1)
sns.set_style("whitegrid")
plt.title("Spending score VS Annual score")
sns.boxplot(y=df["Spending Score (1-100)"], color="red")
plt.subplot(1,2,2)
sns.boxplot(y=df["Annual Income (k$)"])
plt.show()

#7.elbow method to find the value of K 
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
sns.set_style("dark")    
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="blue", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

#8.3d visualization
km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:,1:])
df["label"] = clusters
fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)
ax.scatter(df.Age[df.label == 5], df["Annual Income (k$)"][df.label == 5], df["Spending Score (1-100)"][df.label == 5], c='cyan', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

 #9.2D visualization (scatterplot)
plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label",  
                 palette=['blue','orange','pink','yellow','purple'], legend='full',data = df  ,s = 60 )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()
