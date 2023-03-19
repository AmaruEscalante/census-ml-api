import requests

API_URL = "https://census-ml-api.onrender.com"


def main():
    body = {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": "2174",
        "capital-loss": "0",
        "hours-per-week": "40",
        "native-country": "United-States",
    }
    try:
        response = requests.post(API_URL + "/predict", json=body)
        if response.status_code == 200:
            print(response.json())
        else:
            print("Error: API request failed")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
