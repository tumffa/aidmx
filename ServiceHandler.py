import DataService
import quiet_before_drop
import StatisticsService

dm = DataService.DataManager()
def lesgo(name, file):
    dm.extract_data(name, file)
    segments = StatisticsService.segment(name, dm.get_song(name), ["drums", "other"])
    print("------------------SEGMENTS--------------------")
    for segment in segments:
        print(segment)
lesgo("blue", "blue.mp3")