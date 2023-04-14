import json
from urllib.request import urlopen



def find_score(id_input):
        API_url = "https://1fde-77-201-33-212.ngrok-free.app/credit/" + id_input
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        proba = API_data['score']
        return proba


def test_find_score_with_id():
    
    assert find_score('100001') == 0.53
    assert find_score('100005') == 0.72