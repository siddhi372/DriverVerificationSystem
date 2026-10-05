import serial


class ESP32Manager:

    def __init__(self, port="COM3", baudrate=115200):

        self.port = port
        self.baudrate = baudrate
        self.serial = None

    def connect(self):

        try:
            self.serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0.05
            )

            print("ESP32 connected")

            return True

        except Exception as e:

            print("ESP32 not connected:", e)

            return False

    def read_signal(self):

        if self.serial is None:
            return None

        if self.serial.in_waiting > 0:

            message = (
                self.serial.readline()
                .decode("utf-8")
                .strip()
                .upper()
            )

            return message

        return None

    def send_command(self, command):

        if self.serial is None:
            return

        command = command.upper() + "\n"

        self.serial.write(
            command.encode("utf-8")
        )

        print("Laptop → ESP32:", command.strip())

    def close(self):

        if self.serial is not None:
            self.serial.close()

            print("ESP32 disconnected")