import DataService
import quiet_before_drop
import StatisticsService

dm = DataService.DataManager()
def lesgo(name, file):
    dm.extract_data(name, file)
    segments = StatisticsService.segment(name, dm.get_song(name), ["drums", "other"])
    pauses = quiet_before_drop.get_pauses(name, dm.get_song(name))
    print("------------------PAUSES--------------------")
    print(pauses)
    print("------------------SEGMENTS--------------------")
    for segment in segments:
        print(segment)
lesgo("hail", "hail.mp3")