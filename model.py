import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


user_recommandation=pd.read_csv('data/final_rating.csv')
user_recommandation.set_index('user',inplace=True)
tfid = pickle.load(open('pickle/tfid.pkl','rb'))
randomforest = pickle.load(open('pickle/randomforest_model.pkl','rb'))
data=pd.read_csv('data/processed_sample30.csv')



def predict(user):
    #creating dummy dataframe to store all recommended Products
    recommandation = pd.DataFrame(columns = ['product', 'positive','negative'])
    #checkig if user is existing in recommendation matrix
    if user in user_recommandation.index:
        #getting top 20 Products from recommendation system
        result=user_recommandation.loc[user].sort_values(ascending=False)[0:20]
        prod=data[data['name'].isin(result.index)][['name','processed_text']]
        #passing all 20 products to sentiment model
        for product in prod['name'].unique():
            df=prod[prod['name']==product][['name','processed_text']]
            x_test=df['processed_text']
            X_TFID=tfid.transform(x_test).toarray()
            prob=randomforest.predict_proba(X_TFID)
            recommandation=recommandation.append({'product' : product, 'positive' : round(prob[:,1].mean()*100,2), 'negative' : round(prob[:,0].mean()*100,2)}, 
                      ignore_index = True)

        #getting top 5 products from sentiment model 
        final_recommandation=recommandation[recommandation.positive > recommandation.negative].sort_values('positive',ascending=False)[0:5]
        return True,final_recommandation
    else:
        final_recommandation=''
        return False,final_recommandation

