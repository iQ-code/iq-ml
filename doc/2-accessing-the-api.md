# 2. Accessing the API

## 2.1. Authentication

All API calls require an **API key** passed as an HTTP header:

```
Ocp-Apim-Subscription-Key: <your-api-key>
```

Contact [support@inspiration-q.com](mailto:support@inspiration-q.com) to obtain your API key.

---

## 2.2. Basic Usage with cURL

### Sparse Linear Regression

```sh
curl -X POST https://www.inspiration-q.com/api/v1/iq-ml/linear-regression \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Ocp-Apim-Subscription-Key: <your-api-key>" \
     -d '{
  "X": [
    [1.2, -0.5,  0.3,  2.1, -1.0],
    [0.4,  1.8, -0.7,  0.9,  0.2],
    [-1.1, 0.6,  1.4, -0.3,  1.7],
    [0.8, -1.2,  0.5,  1.6, -0.8],
    [1.5,  0.3, -1.1,  0.7,  1.2]
  ],
  "y": [2.5, 0.9, -1.3, 1.8, 3.1],
  "k": 2,
  "lambda_l2": 0.01,
  "random_number_generator_seed": 42,
  "description": "Sparse linreg example"
}'
```

You will receive a response like:
```json
{"computationId":"4cfa2fc9-85c5-429f-bc9f-a29b1e907763","status":"Computing","computationStoreTimeUtc":"2025-03-04T16:18:03Z"}
```

Then poll for the result:
```sh
curl -X GET "https://www.inspiration-q.com/api/v1/iq-ml/linear-regression/{computationId}" \
  -H "Ocp-Apim-Subscription-Key: <your-api-key>"
```

Once complete:
```json
{"computationId":"4cfa2fc9-85c5-429f-bc9f-a29b1e907763","status":"Ok","computationTimeInSeconds":1.84,"solution":[0.0,-0.0,0.0,1.21,-0.85],"cost":0.043}
```

### Sparse Logistic Regression

```sh
curl -X POST https://www.inspiration-q.com/api/v1/iq-ml/logistic-regression \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Ocp-Apim-Subscription-Key: <your-api-key>" \
     -d '{
  "X": [
    [ 1.2, -0.5,  0.3,  2.1],
    [ 0.4,  1.8, -0.7,  0.9],
    [-1.1,  0.6,  1.4, -0.3],
    [ 0.8, -1.2,  0.5,  1.6]
  ],
  "y": [1, -1, -1, 1],
  "k": 2,
  "description": "Sparse logreg example"
}'
```

### Sparse FDR Regression

Replace the endpoint with `v1/iq-ml/sparse-fdr-regression`; the payload format is identical to logistic regression.

---

## 2.3. Using Python Without the SDK

```python
import requests
import time

BASE_URL = "https://www.inspiration-q.com/api/v1"
API_KEY = "YOUR_API_KEY"
HEADERS = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": API_KEY
}

payload = {
    "X": [
        [ 1.2, -0.5,  0.3,  2.1, -1.0],
        [ 0.4,  1.8, -0.7,  0.9,  0.2],
        [-1.1,  0.6,  1.4, -0.3,  1.7],
        [ 0.8, -1.2,  0.5,  1.6, -0.8],
        [ 1.5,  0.3, -1.1,  0.7,  1.2],
    ],
    "y": [2.5, 0.9, -1.3, 1.8, 3.1],
    "k": 2,
    "lambda_l2": 0.0,
    "random_number_generator_seed": 42,
    "description": "Sparse linreg without SDK"
}

# Step 1: Submit the computation
response = requests.post(
    f"{BASE_URL}/iq-ml/linear-regression", json=payload, headers=HEADERS
)

if response.status_code == 201:
    computation_id = response.json()["computationId"]
    print(f"Computation created. ID: {computation_id}")
else:
    print(f"Error: {response.status_code} - {response.text}")
    exit()

# Step 2: Poll for the result
print("Waiting for result...")
while True:
    response = requests.get(
        f"{BASE_URL}/iq-ml/linear-regression/{computation_id}", headers=HEADERS
    )
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        if result["status"] in ["Ok", "Failed"]:
            print("Result:", result)
            break
    time.sleep(2)
```

The same polling pattern applies to `logistic-regression` and `sparse-fdr-regression`.

---

## 2.4. Computation Lifecycle

All iQ-ML computations follow the same asynchronous lifecycle:

1. **POST** the payload → receive a `computationId` and `status: "Computing"`.
2. **GET** `{endpoint}/{computationId}` → poll until `status` is `"Ok"` or `"Failed"`.
3. Read `solution` and `cost` (linear regression) or `solution` and `weights` (logistic / FDR) from the final response.

The SDK handles steps 2 and 3 automatically.
