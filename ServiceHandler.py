import DataService
import quiet_before_drop

dm = DataService.DataManager()
dm.extract_data("blue", "blue.mp3")
print(dm)
quiet_before_drop.get_pauses("blue", dm.get_song("blue"))