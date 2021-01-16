#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import json

# URL for the web service
scoring_uri = aci_service.swagger_uri #'<your web service URI>'
# If the service is authenticated, set the key or token
key =  aci_service.get_keys()[0]  #'<your key or token>'

# Two sets of data to score, so we get two results back
data = {"data":
        [
            {
               'RI': 5,
               'Na': 2,
               'Mg': 3.5,
               'Al': 0.7,
               'Si': 83,
               'K': 1.2,
               'Ca': 9,
               'Ba': 0.12,
               'Fe': 0.15
               
               },
            {
               'RI': 2,
               'Na': 25,
               'Mg': 2.5,
               'Al': 1.7,
               'Si': 63,
               'K': 2.2,
               'Ca': 6,
               'Ba': 0.15,
               'Fe': 0.18   
           }
        ]
        }
# Convert to JSON string
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)

