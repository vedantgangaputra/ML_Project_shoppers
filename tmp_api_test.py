from fastapi.testclient import TestClient
from api.static.main import app

client = TestClient(app)

response = client.post(
    "/predict",
    json={
        'Administrative': 2,
        'Administrative_Duration': 80.0,
        'Informational': 1,
        'Informational_Duration': 25.0,
        'ProductRelated': 15,
        'ProductRelated_Duration': 450.0,
        'BounceRates': 0.02,
        'ExitRates': 0.05,
        'PageValues': 25.5,
        'SpecialDay': 0.0,
        'Month': 'Aug',
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 1,
        'TrafficType': 2,
        'VisitorType': 'Returning_Visitor',
        'Weekend': 0
    }
)
print(response.status_code)
print(response.json())
