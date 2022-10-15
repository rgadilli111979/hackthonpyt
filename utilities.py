
import pandas as pd
import mysql.connector

def getPredictionData(policyId):
  qry = '''select 
  policy.system_id as id, 
  payment.payment_amount, 
  payment.payment_method,
  DATEDIFF(CURDATE(), policy.policy_effective_date) as age_in_days,
  customer.customer_income as Income,
  policy.uwscore as application_underwriting_score,
  policy.channel_id as sourcing_channel,
  policy.area_type as residence_area_type,
  policy.premium_amount premium,
  payment.days_late
  from policy
  join customer on policy.customer_id = customer.customer_id
  join payment on policy.system_id = payment.system_id
  where policy.system_id = %s'''

  mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Root123$",
    database="prediction"
  )

  curs = mydb.cursor()
  inputs = [policyId]

  curs.execute(qry,inputs)
  #print(curs.column_names)

  rec = {}
  nprem = 0
  cashprem=0
  totprem=0.0000001
  late3 = 0
  late6 = 0
  late12 = 0

  for x in curs:

      #print (x)
      rec['id'] = x[0]
      rec['age_in_days'] = x[3]
      rec['Income'] = x[4]
      rec['application_underwriting_score'] = x[5]
      rec['sourcing_channel'] = x[6]
      rec['residence_area_type'] = x[7]
      rec['premium'] = x[8]
      nprem = nprem + 1
      totprem = totprem + x[1]

      if x[2] == 'cash':
        cashprem = cashprem + x[1]

      dl = int(x[9])
      if dl >= 365:
        late12 = late12 + 1
      elif dl >=180 and dl < 365:
        late6 = late6 + 1
      elif dl >= 30 and dl < 180:
        late3 = late3 + 1
  
  rec['Count_3-6_months_late'] = late3
  rec['Count_6-12_months_late'] = late6
  rec['Count_more_than_12_months_late'] = late12
      

  rec['no_of_premiums_paid'] = nprem
  rec['perc_premium_paid_by_cash_credit'] = cashprem/totprem

  return rec

rslt = getPredictionData(41492)
print(rslt)