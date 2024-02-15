from QueueService import QueueManager

queuemanager = QueueManager("Newsetup.qxw")
# queuemanager.analyze_queue("/mnt/e/MESHUGGAAAH")
name = "jvg"
queuemanager.analyze_track(name, f"./songs/{name}.mp3")