from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
import cv2  # Make sure to import cv2 for OpenCV functions
import argparse
import numpy as np
import datetime
import socket
import sys


class getPulseApp(object):
    """
    Python application that finds a face in a webcam stream, isolates the forehead,
    gathers the average green-light intensity in the forehead region over time,
    and estimates the detected person's pulse.
    """

    def __init__(self, args):
        # Handle serial and UDP connections
        serial = args.serial
        baud = args.baud
        self.send_serial = False
        self.send_udp = False
        if serial:
            self.send_serial = True
            baud = int(baud) if baud else 9600
            self.serial = Serial(port=serial, baudrate=baud)

        udp = args.udp
        if udp:
            self.send_udp = True
            ip, port = (udp.split(":") + [5005])[:2]  # Defaults port to 5005 if not provided
            port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP socket

        # Initialize cameras
        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # First camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.w,self.h = 0, 0
        self.pressed = 0

        # Initialize processor for image analysis
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Initialize parameters for plotting data
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to methods
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv}

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam = (self.selected_cam + 1) % len(self.cameras)

    def write_csv(self):
        """
        Writes current data to a CSV file.
        """
        fn = "Webcam-pulse_" + str(datetime.datetime.now()).replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data, delimiter=',')
        print("Writing CSV to", fn)

    def toggle_search(self):
        """
        Toggles the motion lock on the processor's face detection.
        """
        state = self.processor.find_faces_toggle()
        print("Face detection lock =", not state)
    def toggle_display_plot(self):
        if self.bpm_plot:
            print("BPM plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("BPM plot enabled")
            if self.processor.find_faces:
               self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()

        # Ensure the window exists before moving it
            if cv2.getWindowProperty(self.plot_title, cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow(self.plot_title)
            cv2.moveWindow(self.plot_title, self.w, 0)  # Ensure the window is moved using cv2.moveWindow


    def make_bpm_plot(self):
        """
        Creates or updates the BPM plot display.
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        """
        Handles keystrokes as set in the __init__() method.
        """
        self.pressed = waitKey(10) & 255  # Wait for keypress for 10 ms
        if self.pressed == 27:  # Exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            if self.send_serial:
                self.serial.close()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _ = frame.shape

        # Set current image frame to the processor's input and run analysis
        self.processor.frame_in = frame
        self.processor.run(self.selected_cam)
        output_frame = self.processor.frame_out

        # Show the processed output frame
        imshow("Processed", output_frame)

        # Create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        # Send data via serial or UDP if configured
        if self.send_serial:
            self.serial.write(f"{self.processor.bpm}\r\n".encode())

        if self.send_udp:
            self.sock.sendto(str(self.processor.bpm).encode(), self.udp)

        # Handle any key presses
        self.key_handler()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('--serial', default=None,
                        help='serial port destination for BPM data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='UDP address:port destination for BPM data')

    args = parser.parse_args()
    App = getPulseApp(args)
    while True:
        App.main_loop()
