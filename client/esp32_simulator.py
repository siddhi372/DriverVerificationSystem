class ESP32Simulator:

    def __init__(self):
        self.running = False

    def read_signal(self, key):

        if key == ord("s"):

            print("ESP32 SIMULATOR → START")

            return "START"

        elif key == ord("x"):

            print("ESP32 SIMULATOR → STOP")

            return "STOP"

        return None

    def send_command(self, command):

        print(
            "LAPTOP → ESP32 SIMULATOR:",
            command
        )