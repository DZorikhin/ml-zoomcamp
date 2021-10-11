import pickle

model_file = 'model1.bin'
vect_file = 'dv.bin'

with open(vect_file, 'rb') as v_in:
    dv = pickle.load(v_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

def predict():
    customer = {
        "contract": "two_year", 
        "tenure": 12, 
        "monthlycharges": 19.7
    }

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return result

if __name__ == '__main__':
    print(predict())