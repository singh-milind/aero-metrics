from src.city_info import city_info
CITY_TO_REGION = {
    city: info["region"]
    for city, info in city_info.items()
}