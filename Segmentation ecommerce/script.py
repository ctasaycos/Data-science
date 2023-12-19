'''
Segment the users of the website depending of the behaviour surfing on the web, for this project I decided to use SMOTE for the sampling, because the data is ver unbalanced
it means the users who make the purchases are by far lower than the ones who make it, is a smart idea use this method.(https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
Library: sklearn
Machine learning model: LogisticRegression
Sampling methodology: SMOTE
OS: Mac OS X
Note: This projectas was awarded and invited internationall because was applied in a real company successfully increasing the revenue for over 100% and create a new business line in the consulting firm.
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import metrics
from imblearn.over_sampling import SMOTE

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('/Users/carlostasaycosilva/Library/Mobile Documents/com~apple~CloudDocs/titulacion/modelo/thogarfinal9.csv', header = 0)
print(data.shape)
print(list(data.columns))


#Data Exploration
data['Transaccion'].value_counts()
sns.countplot(x='Transaccion', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')

count_no_sub = len(data[data['Transaccion']==0])
count_sub = len(data[data['Transaccion']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("porcentaje de no transacciones", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("porcentaje de transacciones", pct_of_sub*100)


#Visualization
pd.crosstab(data.canal,data.Transaccion).plot(kind='bar')
plt.title('Frecuencia de transacciones por canal')
plt.xlabel('canal')
plt.ylabel('Frecuencia de transacciones')
plt.savefig('transacciones_por_canal')
plt.show()

table=pd.crosstab(data.tipo_dispositivo,data.Transaccion)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Grafico de barras tipo de dispositivo vs Transacciones')
plt.xlabel('Tipo de dispositivo')
plt.ylabel('Proporcion de visitas')
plt.savefig('transacciones por tipo de dispositivo')
plt.show()

table=pd.crosstab(data.tipo_so,data.Transaccion)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Grafico de barras tipo de sistema operativo vs Transacciones')
plt.xlabel('Tipo de sistema operativo')
plt.ylabel('Proporcion de visitas')
plt.savefig('transacciones por tipo de dispositivo')
plt.show()

table=pd.crosstab(data.Nuevo,data.Transaccion)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Grafico de barras Usuario Nuevo vs Transacciones')
plt.xlabel('Usuario Nuevo')
plt.ylabel('Proporcion de visitas')
plt.savefig('transacciones por tipo de dispositivo')
plt.show()

table=pd.crosstab(data.landing_page,data.Transaccion)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Grafico de barras Pagina de destino vs Transacciones')
plt.xlabel('Pagina de destino')
plt.ylabel('Proporcion de visitas')
plt.savefig('transacciones por tipo de dispositivo')
plt.show()


#Create Dummy Variables
data.head
data_final=data[['Transaccion','menos_30','de_30_60','de_60_mas','nuevaVisita','SEO',   'Campana',  'SocialMedia',  'Email',    'Referencia',   'Directo',  'OtroCanal', 'Mobile',  'Desktop',  'Tablet',   'Android',  'Windows',  'iOS',  'Linux',    'WindowsPhone', 'OtroSO',   'home_movistar',    'home_movil',   'home_hogar',   'home_catalogo',    'catalogo_detalle', 'movistar_landing_caeq',    'movistar_landing_porta',   'catalogo_recomendador',    'catalogo_comparador',  'soporte',  'landing_campana',  'movistar_hogar_duos',  'movistar_hogar_trios', 'movistar_hogar_tv',    'movistar_hogar_internet',  'negocios', 'atencion_cliente']]
data_final.head()
#pd.get_dummies(data.a_canal)

cat_vars=['SEO',    'Campana',  'SocialMedia',  'Email',    'Referencia',   'Directo',  'OtroCanal']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

cat_vars=['SEO',    'Campana',  'SocialMedia',  'Email',    'Referencia',   'Directo',  'OtroCanal']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


#Oversampling using SMOTE
X = data_final.loc[:, data_final.columns != 'Transaccion']
y = data_final.loc[:, data_final.columns == 'Transaccion']


os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Transaccion'])
# we can Check the numbers of our data
print("Tamaño del sobremuestreo de la data ",len(os_data_X))
print("Numero de la no transaccion en la data sobremuestreada",len(os_data_y[os_data_y['Transaccion']==0]))
print("Numero de transacciones",len(os_data_y[os_data_y['a_esTransaccion']==1]))
print("Proporcion de la data de no transaccion en la data sobremuestreada es",len(os_data_y[os_data_y['Transaccion']==0])/len(os_data_X))
print("Proporcion de la data de transaccion en la data sobremuestreada es ",len(os_data_y[os_data_y['Transaccion']==1])/len(os_data_X))




X=os_data_X[['nuevaVisita','menos_30','de_30_60','de_60_mas','SEO', 'Campana',  'SocialMedia',  'Email',    'Referencia',   'Directo',  'OtroCanal',    'Mobile',   'Desktop',  'Tablet',   'Android',  'Windows',  'iOS',  'Linux',    'WindowsPhone','home_hogar','movistar_hogar_duos',  'movistar_hogar_trios', 'movistar_hogar_internet']]
y=os_data_y['Transaccion']


#Data Exploration
os_data_y['Transaccion'].value_counts()
sns.countplot(x='Transaccion', data=os_data_y, palette='hls')
plt.show()
plt.savefig('count_plot')

#Implementing model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


#The p-values for most of the variables are smaller than 0.05, except four variables, 
#therefore, we will remove them.
cols=['nuevaVisita','menos_30','de_30_60','de_60_mas','SEO',    'Campana',  'SocialMedia',  'Email',    'Referencia',   'Directo',  'OtroCanal',    'Mobile',   'Desktop',  'Tablet',   'Linux',    'WindowsPhone','home_hogar','movistar_hogar_duos',  'movistar_hogar_trios', 'movistar_hogar_internet']
X=os_data_X[cols]
y=os_data_y['Transaccion']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#Logistic Regression Model Fitting


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:,1]
print('Precisión del clasificador de regresión logística en el conjunto de prueba: {:.2f}'.format(logreg.score(X_test, y_test)))


y_prob = logreg.predict_proba(X_test)[:,1]
np.percentile(y_prob,50)
np.percentile(y_prob,75)

y_prob = logreg.predict_proba(X_test)[:,1]
plt.boxplot(y_prob)
plt.show()



#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ROC Curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.00])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
