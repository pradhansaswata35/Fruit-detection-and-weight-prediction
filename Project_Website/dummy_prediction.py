# dummy_prediction.py

def get_dummy_prediction():
    return {
        "status": "success",
        "data": [
            {"class": "Apple", "height": 120, "width": 100, "color": "Red", "weight": 160},
            {"class": "Banana", "height": 180, "width": 60, "color": "Yellow", "weight": 120},
            {"class": "Orange", "height": 100, "width": 90, "color": "Orange", "weight": 140},
        ],
        "total_weight": 420
    }
