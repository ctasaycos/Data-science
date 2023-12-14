'''
Customer health score: to classify our customers in the CRM we need the score for priorizing what are the customer agent must call first and set up meetings with them.
CRM: Salesforce
External Data: Google sheets
Machine learning library: sklearn
Machine learning model: LogisticRegression
Delivery: Customer success power bi report
'''

# Import necessary libraries
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
from decouple import config
import pandasql as ps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set Google Sheets API credentials
SERVICE_ACCOUNT_FILE = '/path/customer_health_score.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Connect to Google Sheets API
SAMPLE_SPREADSHEET_ID = 'token'
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="Datos!A1:L500").execute()
data_google = result['values']
df = pd.DataFrame(data_google[1:], columns=data_google[0]).rename(columns=data_google[0])

# Salesforce connection
from simple_salesforce import Salesforce
sf = Salesforce(
    instance_url='https://tt.lightning.force.com/',
    session_id=config('SALESFORCE_SESSION_ID'),
    username='ctasayco@tt.com.pe',
    password='password',
    security_token='sec_token'
)

# Query Salesforce data
opportunity_output = sf.bulk.Opportunity.query("Select Id, N_Oportunidad__c, AccountId, StageName, CloseDate FROM Opportunity where StageName='order/Contrato Recepcionado'")
df_opportunities_bought = pd.DataFrame(opportunity_output, columns=['Id', 'N_Oportunidad__c', 'AccountId', 'StageName', 'CloseDate'])
# ... (similarly query other Salesforce objects)

# Merge Salesforce tables
df_opportunity_bought_product1 = df_opportunities_bought.merge(df_OpportunityLineItem_output.rename({'OpportunityId': 'OpportunityId_r'}, axis=1), left_on='Id', right_on='OpportunityId_r', how='left')
# ... (similarly merge other Salesforce tables)

# Perform analysis and create the final dataset
# ...

# Variable Y
q_scorey = """
    SELECT AccountId, MAX(Closedate) Closedate, CURRENT_DATE, COUNT(DISTINCT Id) AS nro_oport
    FROM df_opportunities_bought0
    GROUP BY AccountId
"""
df_scorey_v1 = ps.sqldf(q_scorey, locals())
df_scorey_v1['Closedate'] = pd.to_datetime(df_scorey_v1['Closedate'])
df_scorey_v1['current_date'] = pd.to_datetime(df_scorey_v1['current_date'])
df_scorey_v1.reset_index(drop=True)
df_scorey_v1['dif'] = (df_scorey_v1['current_date'] - df_scorey_v1['Closedate']).dt.days
q_scorey2 = """
    SELECT AccountId, Closedate, current_date, nro_oport, dif,
    CASE WHEN nro_oport > 1 AND dif < 365 THEN 1 ELSE 0 END 'Y'
    FROM df_scorey_v1
"""
df_scorey_v2 = ps.sqldf(q_scorey2, locals())
df_scorey_v2.head()

# Filter and merge datasets
df = df.loc[(df['number de client'] != 'Nuevo') & (df['number de client'] != 'NUEVO')]
dataset2 = df.merge(df_Account_output.rename({'number_de_client__c': 'number_de_client__c'}, axis=1), left_on='number de client', right_on='number_de_client__c', how='left')
dataset3 = dataset2.merge(df_number_licenses.rename({'AccountId': 'Id2'}, axis=1), left_on='Id', right_on='Id2', how='left')
dataset4 = dataset3.merge(df_number_sspp.rename({'AccountId': 'Id2'}, axis=1), left_on='Id', right_on='Id2', how='left')
dataset5 = dataset4.merge(df_number_cursos.rename({'AccountId': 'Id2'}, axis=1), left_on='Id', right_on='Id2', how='left')
dataset6 = dataset5.merge(df_scorey_v2.rename({'AccountId': 'Id3'}, axis=1), left_on='Id', right_on='Id3', how='left')
dataset6.to_csv('/path/Python/compare1.csv', index=False)

# Modeling
data = dataset6[['Id', 'Flag_sr_Training', 'Flag_cg_Pro', 'Flag_estado', 'perc_creditos', 'perc_usu_activos', 'soporte', 'perc_usopro', 'flag_licenses', 'flag_SSPP', 'flag_cursos', 'Y']]
data = data.fillna(0)
data[['perc_creditos']] = data[['perc_creditos']].stack().str.replace(',', '.').unstack()
data[['perc_usu_activos']] = data[['perc_usu_activos']].stack().str.replace(',', '.').unstack()
data[['perc_usopro']] = data[['perc_usopro']].stack().str.replace(',', '.').unstack()
data["perc_creditos"] = data.perc_creditos.astype(float)
data["perc_usu_activos"] = data.perc_usu_activos.astype(float)
data["soporte"] = data.soporte.astype(float)
data["perc_usopro"] = data.perc_usopro.astype(float)
data["Flag_sr_Training"] = data.Flag_sr_Training.astype(float)
data["Flag_estado"] = data.Flag_estado.astype(float)
data["Flag_cg_Pro"] = data.Flag_cg_Pro.astype(float)
data.dtypes
data = data.set_index("Id", inplace=False)
data_x = data[['Flag_sr_Training', 'Flag_cg_Pro', 'Flag_estado', 'perc_creditos', 'perc_usu_activos', 'soporte', 'perc_usopro', 'flag_licenses', 'flag_SSPP', 'flag_cursos']]
data_y = data[['Y']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=0)

# Logistic Regression model
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

# Model evaluation
score = logisticRegr.score(x_test, y_test)
print(score)

# Model predictions
predictions = logisticRegr.predict(x_test)

# Visualize the correlation matrix
data.dtypes
data['soporte']
np.random.seed(0)
sns.set_theme()
correlation_mat = data.corr()
sns.heatmap(correlation_mat)
plt.show()

# Predicted probabilities
y_pred_proba = logisticRegr.predict_proba(x_test)[::, 1]
