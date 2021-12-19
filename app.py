
 

from flask import Flask, render_template, request

import model
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html',message="")



@app.route('/recommand', methods=['POST'])
def recommand():
	#getting user name from form
    user=request.form['User']
    #predicting top 5 product
    check_user,final_recommandation=model.predict(user)
    if check_user:
        res = render_template('index.html', prediction_text='Product Recommendation for user : {}'.format(user),
                                    tables=[final_recommandation.to_html(classes='data')], titles=final_recommandation.columns.values)
        return res

    else:
    	#if user not present return invalid message
        return render_template('index.html', message="Please enter the valid user")


    


if __name__ == '__main__':
    app.run(debug=True)








