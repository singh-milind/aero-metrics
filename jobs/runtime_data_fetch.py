from data.runtime_data.pm_fetch import main as fetch_pm_data
from data.runtime_data.weather_fetch import main as fetch_weather_data


if __name__ == "__main__":
    fetch_pm_data()
    fetch_weather_data()