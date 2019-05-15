"""
tello_openpose.py : Use the Tello drone as an "selfie air stick"
Relies on tellopy (for interaction with the Tello drone) and Openpose (for body detection and pose recognition)

I started from: https://github.com/Ubotica/telloCV/blob/master/telloCV.py 

"""
import time
import datetime
import os
import tellopy
import numpy as np
import av
import cv2
from pynput import keyboard
import argparse

from math import pi, atan2
from OP import *
from math import atan2, degrees, sqrt
from simple_pid import PID
from  multiprocessing import Process, Pipe, sharedctypes
from FPS import FPS
from CameraMorse import CameraMorse, RollingGraph
from SoundPlayer import SoundPlayer, Tone
import logging
import re
import sys




log = logging.getLogger("TellOpenpose")



def distance (A, B):
    """
        Calculate the square of the distance between points A and B
    """
    return int(sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2))

def angle (A, B, C):
    """
        Calculate the angle between segment(A,p2) and segment (p2,p3)
    """
    if A is None or B is None or C is None:
        return None
    return degrees(atan2(C[1]-B[1],C[0]-B[0]) - atan2(A[1]-B[1],A[0]-B[0]))%360

def vertical_angle (A, B):
    """
        Calculate the angle between segment(A,B) and vertical axe
    """
    if A is None or B is None:
        return None
    return degrees(atan2(B[1]-A[1],B[0]-A[0]) - pi/2)

def quat_to_yaw_deg(qx,qy,qz,qw):
    """
        Calculate yaw from quaternion
    """
    degree = pi/180
    sqy = qy*qy
    sqz = qz*qz
    siny = 2 * (qw*qz+qx*qy)
    cosy = 1 - 2*(qy*qy+qz*qz)
    yaw = int(atan2(siny,cosy)/degree)
    return yaw

def openpose_worker():
    """
        In 2 processes mode, this is the init and main loop of the child
    """
    print("Worker process",os.getpid())
    tello.drone.start_recv_thread()
    tello.init_sounds()
    tello.init_controls()
    tello.op = OP(number_people_max=1, min_size=25, debug=tello.debug)

    while True:
        tello.fps.update()

        frame = np.ctypeslib.as_array(tello.shared_array).copy()
        frame.shape=tello.frame_shape
        
        frame = tello.process_frame(frame)

        cv2.imshow("Processed", frame)

        tello.sound_player.play()
        cv2.waitKey(1)

def main(use_multiprocessing=False, log_level=None):
    """ 
        Create and run a tello controller :
        1) get the video stream from the tello
        2) wait for keyboard commands to pilot the tello
        3) optionnally, process the video frames to track a body and pilot the tello accordingly.

        If use_multiprocessing is True, the parent process creates a child process ('worker')
        and the workload is shared between the 2 processes.
        The parent process job is to:
        - get the video stream from the tello and displays it in an OpenCV window,
        - write each frame in shared memory at destination of the child, 
        each frame replacing the previous one (more efficient than a pipe or a queue),
        - read potential command from the child (currently, only one command:EXIT).
        Commands are transmitted by a Pipe.
        The child process is responsible of all the others tasks:
        - process the frames read in shared memory (openpose, write_hud),
        - if enable, do the tracking (calculate drone commands from position of body),
        - read keyboard commands,
        - transmit commands (from tracking or from keyboard) to the tello, and receive message from the tello.

    """
    global tello
    
    if use_multiprocessing:
        # Create the pipe for the communication between the 2 processes
        parent_cnx, child_cnx = Pipe()
    else:
        child_cnx = None

    tello = TelloController(use_face_tracking=True, 
                            kbd_layout="AZERTY", 
                            write_log_data=False, 
                            log_level=log_level, child_cnx=child_cnx)
   
    first_frame = True  
    frame_skip = 300

    for frame in tello.container.decode(video=0):
        if 0 < frame_skip:
            frame_skip = frame_skip - 1
            continue
        start_time = time.time()
        if frame.time_base < 1.0/60:
            time_base = 1.0/60
        else:
            time_base = frame.time_base

        
        # Convert frame to cv2 image
        frame = cv2.cvtColor(np.array(frame.to_image(),dtype=np.uint8), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640,480))
        if use_multiprocessing:
            
            if first_frame:
                
                # Create the shared memory to share the current frame decoded by the parent process 
                # and given to the child process for further processing (openpose, write_hud,...)
                frame_as_ctypes = np.ctypeslib.as_ctypes(frame)
                tello.shared_array = sharedctypes.RawArray(frame_as_ctypes._type_, frame_as_ctypes)
                tello.frame_shape = frame.shape
                first_frame = False
                # Launch process child
                p_worker = Process(target=openpose_worker)
                p_worker.start()
            # Write the current frame in shared memory
            tello.shared_array[:] = np.ctypeslib.as_ctypes(frame.copy())
            # Check if there is some message from the child
            if parent_cnx.poll():
                msg = parent_cnx.recv()
                if msg == "EXIT":
                    print("MAIN EXIT")
                    p_worker.join()
                    tello.drone.quit()
                    cv2.destroyAllWindows()
                    exit(0)
        else:
            frame = tello.process_frame(frame)
            tello.sound_player.play()

        if not use_multiprocessing: tello.fps.update()

        # Display the frame
        cv2.imshow('Tello', frame)

        cv2.waitKey(1)

        frame_skip = int((time.time() - start_time)/time_base)
    

class TelloController(object):
    """
    TelloController builds keyboard controls on top of TelloPy as well
    as generating images from the video stream and enabling opencv support
    """

    def __init__(self, use_face_tracking=True, 
                kbd_layout="QWERTY", 
                write_log_data=False, 
                media_directory="media", 
                child_cnx=None,
                log_level=None):
        
        self.log_level = log_level
        self.debug = log_level is not None
        self.child_cnx = child_cnx
        self.use_multiprocessing = child_cnx is not None
        self.kbd_layout = kbd_layout
        # Flight data
        self.is_flying = False
        self.battery = None
        self.fly_mode = None
        self.throw_fly_timer = 0

        self.tracking_after_takeoff = False
        self.record = False
        self.keydown = False
        self.date_fmt = '%Y-%m-%d_%H%M%S'
        self.drone = tellopy.Tello(start_recv_thread=not self.use_multiprocessing)
        self.axis_command = {
            "yaw": self.drone.clockwise,
            "roll": self.drone.right,
            "pitch": self.drone.forward,
            "throttle": self.drone.up
        }
        self.axis_speed = { "yaw":0, "roll":0, "pitch":0, "throttle":0}
        self.cmd_axis_speed = { "yaw":0, "roll":0, "pitch":0, "throttle":0}
        self.prev_axis_speed = self.axis_speed.copy()
        self.def_speed =  { "yaw":50, "roll":35, "pitch":35, "throttle":80}     
        
        self.write_log_data = write_log_data
        self.reset()
        self.media_directory = media_directory
        if not os.path.isdir(self.media_directory):
            os.makedirs(self.media_directory)

        if self.write_log_data:
            path = 'tello-%s.csv' % datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            self.log_file = open(path, 'w')
            self.write_header = True

        self.init_drone()
        if not self.use_multiprocessing:
            self.init_sounds()
            self.init_controls()
        

        # container for processing the packets into frames
        self.container = av.open(self.drone.get_video_stream())
        self.vid_stream = self.container.streams.video[0]
        self.out_file = None
        self.out_stream = None
        self.out_name = None
        self.start_time = time.time()

                
        # Setup Openpose
        if not self.use_multiprocessing:
            
            self.op = OP(number_people_max=1, min_size=25, debug=self.debug)
        self.use_openpose = False
                
             
        self.morse = CameraMorse(display=False)
        self.morse.define_command("---", self.delayed_takeoff)
        self.morse.define_command("...", self.throw_and_go, {'tracking':True})
        self.is_pressed = False
       
        self.fps = FPS()

        self.exposure = 0

        if self.debug:
            self.graph_pid = RollingGraph(window_name="PID", step_width=2, width=2000, height=500, y_max=200, colors=[(255,255,255),(255,200,0),(0,0,255),(0,255,0)],thickness=[2,2,2,2],threshold=100, waitKey=False)
                                   

        # Logging
        self.log_level = log_level
        if log_level is not None:
            if log_level == "info":
                log_level = logging.INFO
            elif log_level == "debug":
                log_level = logging.DEBUG
            log.setLevel(log_level)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(log_level)
            ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S"))
            log.addHandler(ch)
        

    def set_video_encoder_rate(self, rate):
        self.drone.set_video_encoder_rate(rate)
        self.video_encoder_rate = rate

    def reset (self):
        """
            Reset global variables before a fly
        """
        log.debug("RESET")
        self.ref_pos_x = -1
        self.ref_pos_y = -1
        self.ref_pos_z = -1
        self.pos_x = -1
        self.pos_y = -1
        self.pos_z = -1
        self.yaw = 0
        self.tracking = False
        self.keep_distance = None
        self.palm_landing = False
        self.palm_landing_approach = False
        self.yaw_to_consume = 0
        self.timestamp_keep_distance = time.time()
        self.wait_before_tracking = None
        self.timestamp_take_picture = None
        self.throw_ongoing = False
        self.scheduled_takeoff = None
        # When in trackin mode, but no body is detected in current frame,
        # we make the drone rotate in the hope to find some body
        # The rotation is done in the same direction as the last rotation done
        self.body_in_prev_frame = False
        self.timestamp_no_body = time.time()
        self.last_rotation_is_cw = True



    def init_drone(self):
        """
            Connect to the drone, start streaming and subscribe to events
        """
        if self.log_level :
            self.drone.log.set_level(2)
        self.drone.connect()
        self.set_video_encoder_rate(2)
        self.drone.start_video()

        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)
        self.drone.subscribe(self.drone.EVENT_LOG_DATA,
                             self.log_data_handler)
        self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED,
                             self.handle_flight_received)

    def init_sounds(self):
        self.sound_player = SoundPlayer(debug=self.debug)
        self.sound_player.load("approaching", "sounds/approaching.ogg")
        self.sound_player.load("keeping distance", "sounds/keeping_distance.ogg")
        self.sound_player.load("landing", "sounds/landing.ogg")
        self.sound_player.load("palm landing", "sounds/palm_landing.ogg")
        self.sound_player.load("taking picture", "sounds/taking_picture.ogg")
        self.sound_player.load("free", "sounds/free.ogg")
        self.tone = Tone()


    def on_press(self, keyname):
        """
            Handler for keyboard listener
        """
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            log.info('KEY PRESS ' + keyname)
            if keyname == 'Key.esc':
                self.toggle_tracking(False)
                # self.tracking = False
                self.drone.land()
                self.drone.quit()
                if self.child_cnx:
                    # Tell to the parent process that it's time to exit
                    self.child_cnx.send("EXIT")
                cv2.destroyAllWindows() 
                os._exit(0)
            if keyname in self.controls_keypress:
                self.controls_keypress[keyname]()
        except AttributeError:
            log.debug(f'special key {keyname0} pressed')

    def on_release(self, keyname):
        """
            Reset on key up from keyboard listener
        """
        self.keydown = False
        keyname = str(keyname).strip('\'')
        log.info('KEY RELEASE ' + keyname)
        if keyname in self.controls_keyrelease:
            key_handler = self.controls_keyrelease[keyname]()
    
    def set_speed(self, axis, speed):
        log.info(f"set speed {axis} {speed}")
        self.cmd_axis_speed[axis] = speed

    def init_controls(self):
        """
            Define keys and add listener
        """

        controls_keypress_QWERTY = {
            'w': lambda: self.set_speed("pitch", self.def_speed["pitch"]),
            's': lambda: self.set_speed("pitch", -self.def_speed["pitch"]),
            'a': lambda: self.set_speed("roll", -self.def_speed["roll"]),
            'd': lambda: self.set_speed("roll", self.def_speed["roll"]),
            'q': lambda: self.set_speed("yaw", -self.def_speed["yaw"]),
            'e': lambda: self.set_speed("yaw", self.def_speed["yaw"]),
            'i': lambda: self.drone.flip_forward(),
            'k': lambda: self.drone.flip_back(),
            'j': lambda: self.drone.flip_left(),
            'l': lambda: self.drone.flip_right(),
            'Key.left': lambda: self.set_speed("yaw", -1.5*self.def_speed["yaw"]),
            'Key.right': lambda: self.set_speed("yaw", 1.5*self.def_speed["yaw"]),
            'Key.up': lambda: self.set_speed("throttle", self.def_speed["throttle"]),
            'Key.down': lambda: self.set_speed("throttle", -self.def_speed["throttle"]),
            'Key.tab': lambda: self.drone.takeoff(),
            'Key.backspace': lambda: self.drone.land(),
            'p': lambda: self.palm_land(),
            't': lambda: self.toggle_tracking(),
            'o': lambda: self.toggle_openpose(),
            'Key.enter': lambda: self.take_picture(),
            'c': lambda: self.clockwise_degrees(360),
            '0': lambda: self.drone.set_video_encoder_rate(0),
            '1': lambda: self.drone.set_video_encoder_rate(1),
            '2': lambda: self.drone.set_video_encoder_rate(2),
            '3': lambda: self.drone.set_video_encoder_rate(3),
            '4': lambda: self.drone.set_video_encoder_rate(4),
            '5': lambda: self.drone.set_video_encoder_rate(5),

            '7': lambda: self.set_exposure(-1),    
            '8': lambda: self.set_exposure(0),
            '9': lambda: self.set_exposure(1)
        }

        controls_keyrelease_QWERTY = {
            'w': lambda: self.set_speed("pitch", 0),
            's': lambda: self.set_speed("pitch", 0),
            'a': lambda: self.set_speed("roll", 0),
            'd': lambda: self.set_speed("roll", 0),
            'q': lambda: self.set_speed("yaw", 0),
            'e': lambda: self.set_speed("yaw", 0),
            'Key.left': lambda: self.set_speed("yaw", 0),
            'Key.right': lambda: self.set_speed("yaw", 0),
            'Key.up': lambda: self.set_speed("throttle", 0),
            'Key.down': lambda: self.set_speed("throttle", 0)
        }

        controls_keypress_AZERTY = {
            'z': lambda: self.set_speed("pitch", self.def_speed["pitch"]),
            's': lambda: self.set_speed("pitch", -self.def_speed["pitch"]),
            'q': lambda: self.set_speed("roll", -self.def_speed["roll"]),
            'd': lambda: self.set_speed("roll", self.def_speed["roll"]),
            'a': lambda: self.set_speed("yaw", -self.def_speed["yaw"]),
            'e': lambda: self.set_speed("yaw", self.def_speed["yaw"]),
            'i': lambda: self.drone.flip_forward(),
            'k': lambda: self.drone.flip_back(),
            'j': lambda: self.drone.flip_left(),
            'l': lambda: self.drone.flip_right(),
            'Key.left': lambda: self.set_speed("yaw", -1.5*self.def_speed["yaw"]),
            'Key.right': lambda: self.set_speed("yaw", 1.5*self.def_speed["yaw"]),
            'Key.up': lambda: self.set_speed("throttle", self.def_speed["throttle"]),
            'Key.down': lambda: self.set_speed("throttle", -self.def_speed["throttle"]),
            'Key.tab': lambda: self.drone.takeoff(),
            'Key.backspace': lambda: self.drone.land(),
            'p': lambda: self.palm_land(),
            't': lambda: self.toggle_tracking(),
            'o': lambda: self.toggle_openpose(),
            'Key.enter': lambda: self.take_picture(),
            'c': lambda: self.clockwise_degrees(360),
            '0': lambda: self.drone.set_video_encoder_rate(0),
            '1': lambda: self.drone.set_video_encoder_rate(1),
            '2': lambda: self.drone.set_video_encoder_rate(2),
            '3': lambda: self.drone.set_video_encoder_rate(3),
            '4': lambda: self.drone.set_video_encoder_rate(4),
            '5': lambda: self.drone.set_video_encoder_rate(5),

            '7': lambda: self.set_exposure(-1),    
            '8': lambda: self.set_exposure(0),
            '9': lambda: self.set_exposure(1)
        }

        controls_keyrelease_AZERTY = {
            'z': lambda: self.set_speed("pitch", 0),
            's': lambda: self.set_speed("pitch", 0),
            'q': lambda: self.set_speed("roll", 0),
            'd': lambda: self.set_speed("roll", 0),
            'a': lambda: self.set_speed("yaw", 0),
            'e': lambda: self.set_speed("yaw", 0),
            'Key.left': lambda: self.set_speed("yaw", 0),
            'Key.right': lambda: self.set_speed("yaw", 0),
            'Key.up': lambda: self.set_speed("throttle", 0),
            'Key.down': lambda: self.set_speed("throttle", 0)
        }

        if self.kbd_layout == "AZERTY":
            self.controls_keypress = controls_keypress_AZERTY
            self.controls_keyrelease = controls_keyrelease_AZERTY
        else:
            self.controls_keypress = controls_keypress_QWERTY
            self.controls_keyrelease = controls_keyrelease_QWERTY
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()

     
    def check_pose(self, w, h):
        """
            Check if we detect a pose in the body detected by Openpose
        """
    
        neck = self.op.get_body_kp("Neck")
        r_wrist = self.op.get_body_kp("RWrist")
        l_wrist = self.op.get_body_kp("LWrist")
        r_elbow = self.op.get_body_kp("RElbow")
        l_elbow = self.op.get_body_kp("LElbow")
        r_shoulder = self.op.get_body_kp("RShoulder")
        l_shoulder = self.op.get_body_kp("LShoulder")
        r_ear = self.op.get_body_kp("REar")
        l_ear = self.op.get_body_kp("LEar") 
        
        self.shoulders_width = distance(r_shoulder,l_shoulder) if r_shoulder and l_shoulder else None


        vert_angle_right_arm = vertical_angle(r_wrist, r_elbow)
        vert_angle_left_arm = vertical_angle(l_wrist, l_elbow)

        left_hand_up = neck and l_wrist and l_wrist[1] < neck[1]
        right_hand_up = neck and r_wrist and r_wrist[1] < neck[1]

        if right_hand_up:
            if not left_hand_up:
                # Only right arm up
                if r_ear and (r_ear[0]-neck[0])*(r_wrist[0]-neck[0])>0:
                # Right ear and right hand on the same side
                    if vert_angle_right_arm:
                        if vert_angle_right_arm < -15:
                            return "RIGHT_ARM_UP_OPEN"
                        if 15 < vert_angle_right_arm < 90:
                            return "RIGHT_ARM_UP_CLOSED"
                elif l_ear and self.shoulders_width and distance(r_wrist,l_ear) < self.shoulders_width/4:
                    # Right hand close to left ear
                    return "RIGHT_HAND_ON_LEFT_EAR"
            else:
                # Both hands up
                # Check if both hands are on the ears
                if r_ear and l_ear:
                    ear_dist = distance(r_ear,l_ear)
                    if distance(r_wrist,r_ear)<ear_dist/3 and distance(l_wrist,l_ear)<ear_dist/3:
                        return("HANDS_ON_EARS")
                # Check if boths hands are closed to each other and above ears 
                # (check right hand is above right ear is enough since hands are closed to each other)
                if self.shoulders_width and r_ear:
                    near_dist = self.shoulders_width/3
                    if r_ear[1] > r_wrist[1] and distance(r_wrist, l_wrist) < near_dist :
                        return "CLOSE_HANDS_UP"

        else:
            if left_hand_up:
                # Only left arm up
                if l_ear and (l_ear[0]-neck[0])*(l_wrist[0]-neck[0])>0:
                    # Left ear and left hand on the same side
                    if vert_angle_left_arm:
                        if vert_angle_left_arm < -15:
                            return "LEFT_ARM_UP_CLOSED"
                        if 15 < vert_angle_left_arm < 90:
                            return "LEFT_ARM_UP_OPEN"
                elif r_ear and self.shoulders_width and distance(l_wrist,r_ear) < self.shoulders_width/4:
                    # Left hand close to right ear
                    return "LEFT_HAND_ON_RIGHT_EAR"
            else:
                # Both wrists under the neck
                if neck and self.shoulders_width and r_wrist and l_wrist:
                    near_dist = self.shoulders_width/3
                    if distance(r_wrist, neck) < near_dist and distance(l_wrist, neck) < near_dist :
                        return "HANDS_ON_NECK"

        return None

    def process_frame(self, raw_frame):
        """
            Analyze the frame and return the frame with information (HUD, openpose skeleton) drawn on it
        """
        
        frame = raw_frame.copy()
        h,w,_ = frame.shape
        proximity = int(w/2.6)
        min_distance = int(w/2)
        
        # Is there a scheduled takeoff ?
        if self.scheduled_takeoff and time.time() > self.scheduled_takeoff:
            
            self.scheduled_takeoff = None
            self.drone.takeoff()

        self.axis_speed = self.cmd_axis_speed.copy()

        # If we are on the point to take a picture, the tracking is temporarily desactivated (2s)
        if self.timestamp_take_picture:
            if time.time() - self.timestamp_take_picture > 2:
                self.timestamp_take_picture = None
                self.drone.take_picture()
        else:

            # If we are doing a 360, where are we in our 360 ?
            if self.yaw_to_consume > 0:
                consumed = self.yaw - self.prev_yaw
                self.prev_yaw = self.yaw
                if consumed < 0: consumed += 360
                self.yaw_consumed += consumed
                if self.yaw_consumed > self.yaw_to_consume:
                    self.yaw_to_consume = 0
                    self.axis_speed["yaw"] = 0
                else:
                    self.axis_speed["yaw"] = self.def_speed["yaw"]

            # We are not flying, we check a potential morse code 
            if not self.is_flying:
                pressing, detected = self.morse.eval(frame)
                if self.is_pressed and not pressing:
                    self.tone.off()
                elif not self.is_pressed and pressing:
                    self.tone.on()
                self.is_pressed = pressing


            # Call to openpose detection
            if self.use_openpose:

                nb_people, pose_kps, face_kps = self.op.eval(frame)
                
                target = None
                
                # Our target is the person whose index is 0 in pose_kps
                self.pose = None
                if nb_people > 0 : 
                    # We found a body, so we can cancel the exploring 360
                    self.yaw_to_consume = 0

                    # Do we recognize a predefined pose ?
                    self.pose = self.check_pose(w,h)

                    if self.pose:
                        # We trigger the associated action
                        log.info(f"pose detected : {self.pose}")
                        if self.pose == "HANDS_ON_NECK" or self.pose == "HANDS_ON_EARS":
                            # Take a picture in 1 second
                            if self.timestamp_take_picture is None:
                                log.info("Take a picture in 1 second")
                                self.timestamp_take_picture = time.time()
                                self.sound_player.play("taking picture")
                        elif self.pose == "RIGHT_ARM_UP_CLOSED":
                            log.info("GOING LEFT from pose")
                            self.axis_speed["roll"] = self.def_speed["roll"]
                        elif self.pose == "RIGHT_ARM_UP_OPEN":
                            log.info("GOING RIGHT from pose")
                            self.axis_speed["roll"] = -self.def_speed["roll"]
                        elif self.pose == "LEFT_ARM_UP_CLOSED":
                            log.info("GOING FORWARD from pose")
                            self.axis_speed["pitch"] = self.def_speed["pitch"]
                        elif self.pose == "LEFT_ARM_UP_OPEN":
                            log.info("GOING BACKWARD from pose")
                            self.axis_speed["pitch"] = -self.def_speed["pitch"]
                        elif self.pose == "CLOSE_HANDS_UP":
                            # Locked distance mode
                            if self.keep_distance is None:
                                if  time.time() - self.timestamp_keep_distance > 2:
                                    # The first frame of a serie to activate the distance keeping
                                    self.keep_distance = self.shoulders_width
                                    self.timestamp_keep_distance = time.time()
                                    log.info(f"KEEP DISTANCE {self.keep_distance}")
                                    self.pid_pitch = PID(0.5,0.04,0.3,setpoint=0,output_limits=(-50,50))
                                    #self.graph_distance = RollingGraph(window_name="Distance", y_max=500, threshold=self.keep_distance, waitKey=False)
                                    self.sound_player.play("keeping distance")
                            else:
                                if time.time() - self.timestamp_keep_distance > 2:
                                    log.info("KEEP DISTANCE FINISHED")
                                    self.sound_player.play("free")
                                    self.keep_distance = None
                                    self.timestamp_keep_distance = time.time()
                            
                        elif self.pose == "RIGHT_HAND_ON_LEFT_EAR":
                            # Get close to the body then palm landing
                            if not self.palm_landing_approach:
                                self.palm_landing_approach = True
                                self.keep_distance = proximity
                                self.timestamp_keep_distance = time.time()
                                log.info("APPROACHING on pose")
                                self.pid_pitch = PID(0.2,0.02,0.1,setpoint=0,output_limits=(-45,45))
                                #self.graph_distance = RollingGraph(window_name="Distance", y_max=500, threshold=self.keep_distance, waitKey=False)
                                self.sound_player.play("approaching")
                        elif self.pose == "LEFT_HAND_ON_RIGHT_EAR":
                            if not self.palm_landing:
                                log.info("LANDING on pose")
                                # Landing
                                self.toggle_tracking(tracking=False)
                                self.drone.land()      

                    # Draw the skeleton on the frame
                    self.op.draw_body(frame)
                    
                    # In tracking mode, we track a specific body part (an openpose keypoint):
                    # the nose if visible, otherwise the neck, otherwise the midhip
                    # The tracker tries to align that body part with the reference point (ref_x, ref_y)
                    target = self.op.get_body_kp("Nose")
                    if target is not None:        
                        ref_x = int(w/2)
                        ref_y = int(h*0.35)
                    else:
                        target = self.op.get_body_kp("Neck")
                        if target is not None:         
                            ref_x = int(w/2)
                            ref_y = int(h/2)
                        else:
                            target = self.op.get_body_kp("MidHip")
                            if target is not None:         
                                ref_x = int(w/2)
                                ref_y = int(0.75*h)
                    

                if self.tracking:
                    if target:
                        self.body_in_prev_frame = True
                        # We draw an arrow from the reference point to the body part we are targeting       
                        h,w,_ = frame.shape
                        xoff = int(target[0]-ref_x)
                        yoff = int(ref_y-target[1])
                        cv2.circle(frame, (ref_x, ref_y), 15, (250,150,0), 1,cv2.LINE_AA)
                        cv2.arrowedLine(frame, (ref_x, ref_y), target, (250, 150, 0), 6)
                       
                        # The PID controllers calculate the new speeds for yaw and throttle
                        self.axis_speed["yaw"] = int(-self.pid_yaw(xoff))
                        log.debug(f"xoff: {xoff} - speed_yaw: {self.axis_speed['yaw']}")
                        self.last_rotation_is_cw = self.axis_speed["yaw"] > 0

                        self.axis_speed["throttle"] = int(-self.pid_throttle(yoff))
                        log.debug(f"yoff: {yoff} - speed_throttle: {self.axis_speed['throttle']}")

                        # If in locke distance mode
                        if self.keep_distance and self.shoulders_width:   
                            if self.palm_landing_approach and self.shoulders_width>self.keep_distance:
                                # The drone is now close enough to the body
                                # Let's do the palm landing
                                log.info("PALM LANDING after approaching")
                                self.palm_landing_approach = False
                                self.toggle_tracking(tracking=False)
                                self.palm_land() 
                            else:
                                self.axis_speed["pitch"] = int(self.pid_pitch(self.shoulders_width-self.keep_distance))
                                log.debug(f"Target distance: {self.keep_distance} - cur: {self.shoulders_width} -speed_pitch: {self.axis_speed['pitch']}")
                    else: # Tracking but no body detected
                        if self.body_in_prev_frame:
                            self.timestamp_no_body = time.time()
                            self.body_in_prev_frame = False
                            self.axis_speed["throttle"] = self.prev_axis_speed["throttle"]
                            self.axis_speed["yaw"] = self.prev_axis_speed["yaw"]
                        else:
                            if time.time() - self.timestamp_no_body < 1:
                                print("NO BODY SINCE < 1", self.axis_speed, self.prev_axis_speed)
                                self.axis_speed["throttle"] = self.prev_axis_speed["throttle"]
                                self.axis_speed["yaw"] = self.prev_axis_speed["yaw"]
                            else:
                                log.debug("NO BODY detected for 1s -> rotate")
                                self.axis_speed["yaw"] = self.def_speed["yaw"] * (1 if self.last_rotation_is_cw else -1)

        # Send axis commands to the drone
        for axis, command in self.axis_command.items():
            if self.axis_speed[axis]is not None and self.axis_speed[axis] != self.prev_axis_speed[axis]:
                log.debug(f"COMMAND {axis} : {self.axis_speed[axis]}")
                command(self.axis_speed[axis])
                self.prev_axis_speed[axis] = self.axis_speed[axis]
            else:
                # This line is necessary to display current values in 'self.write_hud'
                self.axis_speed[axis] = self.prev_axis_speed[axis]
        
        # Write the HUD on the frame
        frame = self.write_hud(frame)

        
        return frame

    def write_hud(self, frame):
        """
            Draw drone info on frame
        """

        class HUD:
            def __init__(self, def_color=(255, 170, 0)):
                self.def_color = def_color
                self.infos = []
            def add(self, info, color=None):
                if color is None: color = self.def_color
                self.infos.append((info, color))
            def draw(self, frame):
                i=0
                for (info, color) in self.infos:
                    cv2.putText(frame, info, (0, 30 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2) #lineType=30)
                    i+=1
                

        hud = HUD()

        if self.debug: hud.add(datetime.datetime.now().strftime('%H:%M:%S'))
        hud.add(f"FPS {self.fps.get():.2f}")
        if self.debug: hud.add(f"VR {self.video_encoder_rate}")

        hud.add(f"BAT {self.battery}")
        if self.is_flying:
            hud.add("FLYING", (0,255,0))
        else:
            hud.add("NOT FLYING", (0,0,255))
        hud.add(f"TRACKING {'ON' if self.tracking else 'OFF'}", (0,255,0) if self.tracking else (0,0,255) )
        hud.add(f"EXPO {self.exposure}")
        
        if self.axis_speed['yaw'] > 0:
            hud.add(f"CW {self.axis_speed['yaw']}", (0,255,0))
        elif self.axis_speed['yaw'] < 0:
            hud.add(f"CCW {-self.axis_speed['yaw']}", (0,0,255))
        else:
            hud.add(f"CW 0")
        if self.axis_speed['roll'] > 0:
            hud.add(f"RIGHT {self.axis_speed['roll']}", (0,255,0))
        elif self.axis_speed['roll'] < 0:
            hud.add(f"LEFT {-self.axis_speed['roll']}", (0,0,255))
        else:
            hud.add(f"RIGHT 0")
        if self.axis_speed['pitch'] > 0:
            hud.add(f"FORWARD {self.axis_speed['pitch']}", (0,255,0))
        elif self.axis_speed['pitch'] < 0:
            hud.add(f"BACKWARD {-self.axis_speed['pitch']}", (0,0,255))
        else:
            hud.add(f"FORWARD 0")
        if self.axis_speed['throttle'] > 0:
            hud.add(f"UP {self.axis_speed['throttle']}", (0,255,0))
        elif self.axis_speed['throttle'] < 0:
            hud.add(f"DOWN {-self.axis_speed['throttle']}", (0,0,255))
        else:
            hud.add(f"UP 0")

        if self.use_openpose: hud.add(f"POSE: {self.pose}", (0,255,0) if self.pose else (255, 170, 0))
        if self.keep_distance: 
            hud.add(f"Target distance: {self.keep_distance} - curr: {self.shoulders_width}", (0,255,0))
            #if self.shoulders_width: self.graph_distance.new_iter([self.shoulders_width])
        if self.timestamp_take_picture: hud.add("Taking a picture", (0,255,0))
        if self.palm_landing:
            hud.add("Palm landing...", (0,255,0))
        if self.palm_landing_approach:
            hud.add("In approach for palm landing...", (0,255,0))
        if self.tracking and not self.body_in_prev_frame and time.time() - self.timestamp_no_body > 0.5:
            hud.add("Searching...", (0,255,0))
        if self.throw_ongoing:
            hud.add("Throw ongoing...", (0,255,0))
        if self.scheduled_takeoff:
            seconds_left = int(self.scheduled_takeoff - time.time())
            hud.add(f"Takeoff in {seconds_left}s")

        hud.draw(frame)
        return frame

    def take_picture(self):
        """
            Tell drone to take picture, image sent to file handler
        """
        self.drone.take_picture()

    def set_exposure(self, expo):
        """
            Change exposure of drone camera
        """
        if expo == 0:
            self.exposure = 0
        elif expo == 1:
            self.exposure = min(9, self.exposure+1)
        elif expo == -1:
            self.exposure = max(-9, self.exposure-1)
        self.drone.set_exposure(self.exposure)
        log.info(f"EXPOSURE {self.exposure}")

    def palm_land(self):
        """
            Tell drone to land
        """
        self.palm_landing = True
        self.sound_player.play("palm landing")
        self.drone.palm_land()

    def throw_and_go(self, tracking=False):
        """
            Tell drone to start a 'throw and go'
        """
        self.drone.throw_and_go()      
        self.tracking_after_takeoff = tracking
        
    def delayed_takeoff(self, delay=5):
        self.scheduled_takeoff = time.time()+delay
        self.tracking_after_takeoff = True
        
    def clockwise_degrees(self, degrees):
        self.yaw_to_consume = degrees
        self.yaw_consumed = 0
        self.prev_yaw = self.yaw
        
    def toggle_openpose(self):
        self.use_openpose = not self.use_openpose
        if not self.use_openpose:
            # Desactivate tracking
            self.toggle_tracking(tracking=False)
        log.info('OPENPOSE '+("ON" if self.use_openpose else "OFF"))

         
    def toggle_tracking(self, tracking=None):
        """ 
            If tracking is None, toggle value of self.tracking
            Else self.tracking take the same value as tracking
        """
        
        if tracking is None:
            self.tracking = not self.tracking
        else:
            self.tracking = tracking
        if self.tracking:
            log.info("ACTIVATE TRACKING")
            # Needs openpose
            self.use_openpose = True
            # Start an explarotary 360
            #self.clockwise_degrees(360)
            # Init a PID controller for the yaw
            self.pid_yaw = PID(0.25,0,0,setpoint=0,output_limits=(-100,100))
            # ... and one for the throttle
            self.pid_throttle = PID(0.4,0,0,setpoint=0,output_limits=(-80,100))
            # self.init_tracking = True
        else:
            self.axis_speed = { "yaw":0, "roll":0, "pitch":0, "throttle":0}
            self.keep_distance = None
        return

    def flight_data_handler(self, event, sender, data):
        """
            Listener to flight data from the drone.
        """
        self.battery = data.battery_percentage
        self.fly_mode = data.fly_mode
        self.throw_fly_timer = data.throw_fly_timer
        self.throw_ongoing = data.throw_fly_timer > 0

        # print("fly_mode",data.fly_mode)
        # print("throw_fly_timer",data.throw_fly_timer)
        # print("em_ground",data.em_ground)
        # print("em_sky",data.em_sky)
        # print("electrical_machinery_state",data.electrical_machinery_state)
        #print("em_sky",data.em_sky,"em_ground",data.em_ground,"em_open",data.em_open)
        #print("height",data.height,"imu_state",data.imu_state,"down_visual_state",data.down_visual_state)
        if self.is_flying != data.em_sky:            
            self.is_flying = data.em_sky
            log.debug(f"FLYING : {self.is_flying}")
            if not self.is_flying:
                self.reset()
            else:
                if self.tracking_after_takeoff:
                    log.info("Tracking on after takeoff")
                    self.toggle_tracking(True)
                    
        log.debug(f"MODE: {self.fly_mode} - Throw fly timer: {self.throw_fly_timer}")

    def log_data_handler(self, event, sender, data):
        """
            Listener to log data from the drone.
        """  
        pos_x = -data.mvo.pos_x
        pos_y = -data.mvo.pos_y
        pos_z = -data.mvo.pos_z
        if abs(pos_x)+abs(pos_y)+abs(pos_z) > 0.07:
            if self.ref_pos_x == -1: # First time we have meaningful values, we store them as reference
                self.ref_pos_x = pos_x
                self.ref_pos_y = pos_y
                self.ref_pos_z = pos_z
            else:
                self.pos_x = pos_x - self.ref_pos_x
                self.pos_y = pos_y - self.ref_pos_y
                self.pos_z = pos_z - self.ref_pos_z
        
        qx = data.imu.q1
        qy = data.imu.q2
        qz = data.imu.q3
        qw = data.imu.q0
        self.yaw = quat_to_yaw_deg(qx,qy,qz,qw)
        
        if self.write_log_data:
            if self.write_header:
                self.log_file.write('%s\n' % data.format_cvs_header())
                self.write_header = False
            self.log_file.write('%s\n' % data.format_cvs())

    def handle_flight_received(self, event, sender, data):
        """
            Create a file in local directory to receive image from the drone
        """
        path = f'{self.media_directory}/tello-{datetime.datetime.now().strftime(self.date_fmt)}.jpg' 
        with open(path, 'wb') as out_file:
            out_file.write(data)
        log.info('Saved photo to %s' % path)

    


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-l","--log_level", help="select a log level (info, debug)")
    ap.add_argument("-2","--multiprocess", action='store_true', help="use 2 processes to share the workload (instead of 1)")
    args=ap.parse_args()

    main(use_multiprocessing=args.multiprocess, log_level=args.log_level)