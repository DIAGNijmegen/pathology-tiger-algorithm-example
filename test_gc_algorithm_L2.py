
import gcapi  # pip install gcapi

your_algorithm_slug = "tiger-algorithm-example"           # <--- CHANGE THIS
client = gcapi.Client(token="")                           # <--- CHANGE THIS, more information about the token please see this link: https://grand-challenge.org/documentation/what-can-gc-api-be-used-for/

job = client.run_external_job(
    algorithm=your_algorithm_slug,
    inputs={
        # 104S (from the tils training subset)
        "generic-medical-image": "https://grand-challenge.org/api/v1/cases/images/18a9e579-34bd-43b7-ac42-61541fb35156/",
        # 104S_tissue (similar mask as expected in L2)
        "generic-overlay": "https://grand-challenge.org/api/v1/cases/images/e61811ac-7080-4f3d-becf-efbc9c39d99e/"
    }
)
# More information about gcapi please see this link: https://grand-challenge.org/documentation/grand-challenge-api/
