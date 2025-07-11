import requests

if __name__ == '__main__':
    url = "https://chat.noc.pku.edu.cn/v1/models"
    headers = {
        "Authorization": "Bearer sk-ZhouYanZhen_xw0k51BG2hHR"
    }

    response = requests.get(url, headers=headers)
    all_usable_models = response.json()["data"]
    for model in all_usable_models:
        # print(f"id: {model['id']}, owned_by: {model['owned_by']}")
        print(f"{model['id']}")