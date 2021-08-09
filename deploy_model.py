from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel
#import numpy as np

import joblib
filename = 'lasso_model.h5'
lasso_model = joblib.load(filename)

# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
	area : float
	floor_number : float
	location_id : float
	project_features : float

# Creating an Endpoint to recieve the data
# to make prediction on.
@app.post('/predict')
def predict(data : request_body):
	# Making the data in a form suitable for prediction
	test_data = [[
			data.area,
			data.floor_number,
			data.location_id,
			data.project_features
	]]
	
	# Predicting the Class
	sale_price = lasso_model.predict(test_data)[0]
	
	# Return the Result
	return { 'sale_price' : sale_price}


