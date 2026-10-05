import requests
import json


API_URL = "http://127.0.0.1:8000/verification/result"


def send_verification_result(verification_data):
    """
    Send verification result from laptop AI application
    to the FastAPI backend.
    """

    try:

        # Show exactly what the AI is sending
        print("\n==========================================")
        print("AI OUTPUT BEING SENT TO BACKEND")
        print("==========================================")
        print(json.dumps(verification_data, indent=4, default=str))
        print("==========================================\n")

        response = requests.post(
            API_URL,
            json=verification_data,
            timeout=5
        )

        # Show backend response
        print("BACKEND STATUS CODE:", response.status_code)
        print("BACKEND RESPONSE:")

        try:
            print(json.dumps(response.json(), indent=4))
        except Exception:
            print(response.text)

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:

        print("\n==========================================")
        print("BACKEND REQUEST FAILED")
        print("ERROR:", e)
        print("==========================================\n")

        raise