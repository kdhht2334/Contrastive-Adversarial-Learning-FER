__author__ = "kdhht5022@gmail.com"
"""
Do make sure: opencv-python: 3.4.1 

using following cmd:
    $pip install opencv-python==3.4.1.0
"""
import numpy as np
from PIL import Image
import time, os
from multiprocessing import Queue, Value
import configparser, uuid

import asyncio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2

from my_util.videomgr import VideoMgr
from my_util.detect_util import draw_results_ssd
from my_util.fer_util import nn_output


async def async_handle_video(camInfo):
    videoSaveOut = None
    config = camInfo["conf"]    
    VideoHandler = VideoMgr(int(config['url']), config['name'])        
    VideoHandler.open(config)
   
    loop = asyncio.get_event_loop()

    global key
    global f
    global image_queue
    global cam_check
    
    global do_FER
    
    global image_batch
    
    global valence, arousal
    global fd_signal
    global orig_image
    
    global emotion_list  # ["angry", "sad", "happy", "pleased", "neutral"]
    global emot_region
    
    def sleep():
        time.sleep(0.02)
        return 
    def resize(img, size):
        return cv2.resize(img, size)
    try:
        f = 0  # just for counting :)
        while(True):
            if cam_check[cname[VideoHandler.camName]] == 0:
                ret, orig_image = await loop.run_in_executor(None, VideoHandler.camCtx.read)
                if cname[VideoHandler.camName]%2 == 1:
                    orig_image = np.fliplr(orig_image)
                    

                # Trigger using some condition
                if f % 2 == 0:
                    do_FER = True
                    if len(image_batch) > 10: 
                        image_batch[:5] = []
                    image_batch.append(orig_image)
                    
                    if type(valence) is torch.Tensor:
                        valence = valence.detach().cpu().numpy()
                        arousal = arousal.detach().cpu().numpy()
                    if np.abs(valence) < 0.1 and np.abs(arousal) < 0.1:
                        final_emot = emotion_list[4]  # neutral
                    elif valence > 0.1 and arousal > 0.2:
                        final_emot = emotion_list[2]  # happy
                    elif valence < -0.2 and arousal > 0.1:
                        final_emot = emotion_list[0]  # angry
                    elif valence < -0.1 and arousal < -0.1:
                        final_emot = emotion_list[1]  # sad
                    elif valence > 0.1 and arousal < -0.1:
                        final_emot = emotion_list[3]  # pleased
                        
                    if np.sign(valence) == 1 and np.sign(arousal) == 1:
                        emot_region = "1R"
                    elif np.sign(valence) == -1 and np.sign(arousal) == 1:
                        emot_region = "2R"
                    elif np.sign(valence) == -1 and np.sign(arousal) == -1:
                        emot_region = "3R"
                    elif np.sign(valence) == 1 and np.sign(arousal) == -1:
                        emot_region = "4R"
                    
                    
                show_img = orig_image.copy()
                if (config['camshow'] == 'on'):# & (cname[VideoHandler.camName]%2 == 0 ) & (False):

                    cv2.putText(show_img, 'Valence: {0:.2f}'.format(np.round(float(valence),2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(show_img, 'Arousal: {0:.2f}'.format(np.round(float(arousal),2)), (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(show_img, '[{}] Emotion: {}'.format(emot_region, final_emot), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if fd_signal == 1:
                        cv2.putText(show_img, 'Face is detected', (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
                        cv2.rectangle(show_img, face_Bbox[0], face_Bbox[1], (255, 77, 9), 3, 1)
                    elif fd_signal == 0:
                        cv2.putText(show_img, 'Face is NOT detected', (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                    cv2.imshow(config['name'], show_img)

                    if VideoHandler.camName == '1th_left':
                        key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                f += 1
                    
            else:
                await loop.run_in_executor(None, sleep)
                if sum(cam_check) == 4:
                    cam_check = [0,0,0,0]


        cv2.destroyAllWindows()
        VideoHandler.camCtx.release()

    except asyncio.CancelledError:        
        if config['tracking_flag'] == 'off' and config['videosave'] == 'on':        
            videoSaveOut.release()
            
        cv2.destroyAllWindows()
        VideoHandler.camCtx.release()


async def handle_video_analysis():    
    def FER():

        global do_FER
        global end_face_detection

        global image_batch
        
        global faces
        global net
        global encoder, regressor, disc
        global f
        
        global valence, arousal
        global fd_signal
        global input_img, img_w, img_h
        global cropped_face
        global face_Bbox

        if do_FER == 1:
            
            input_img = image_batch[-1]
            img_h, img_w, _ = np.shape(input_img)
            
            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detected = net.forward()  # For example, (1, 1, 200, 7)  

            faces = np.empty((detected.shape[2], 224, 224, 3))
            cropped_face, fd_signal, face_Bbox = draw_results_ssd(detected,input_img,faces,0.1,224,img_w,img_h,0,0,0)  # 128
            croppted_face_tr = torch.from_numpy(cropped_face.transpose(0,3,1,2)[0]/255.)  # [3, 224, 224]
            cropped_face_th_norm = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(croppted_face_tr)
    
            latent_feature = encoder(cropped_face_th_norm.unsqueeze_(0).type(torch.FloatTensor ))#cuda
            va_output = regressor(latent_feature)
            
            valence = va_output.detach().cpu().numpy()[0][0] + 0.15
            arousal = va_output.detach().cpu().numpy()[0][1] + 0.15
            

            do_FER = False
            end_face_detection = True
#        else:
#            hand_gesture_sleep = True

    loop = asyncio.get_event_loop()
    try:
        while(True):
            await loop.run_in_executor(None, FER)
            if key == ord('q'):
               break
    except asyncio.CancelledError:        
        pass


# ---------------------
# Capture & store image
# ---------------------
async def store_img():
    def store():
        global cap_count, input_img
        try:
            if fd_signal == 1:
                cv2.imwrite('crop_real/cropped_{}.png'.format(cap_count),
                            cropped_face[0])
            if hand_start[0] != hand_end[0]:
                crop_hand = input_img[hand_start[0][1]:hand_end[0][1], hand_start[0][0]:hand_end[0][0]]
                cv2.imwrite('crop_hand/cropped_{}(1st).png'.format(cap_count),
                            crop_hand)
            if hand_start[1] != hand_end[1]:
                crop_hand = input_img[hand_start[1][1]:hand_end[1][1], hand_start[1][0]:hand_end[1][0]]
                cv2.imwrite('crop_hand/cropped_{}(2nd).png'.format(cap_count),
                            crop_hand)
            cap_count += 1
            print('capture {}'.format(cap_count))
        except:
            print('image store fail...')

    loop = asyncio.get_event_loop()
    try:
        while (True):
            if key == ord('q'):
                break
            await loop.run_in_executor(None, store)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass


# -----------------
# Paralle processor
# -----------------
async def async_handle_video_run(camInfos):  
    futures = [asyncio.ensure_future(async_handle_video(cam_info)) for cam_info in camInfos]\
             +[asyncio.ensure_future(handle_video_analysis())]
    await asyncio.gather(*futures)


class Config():
    """  Configuration for Label Convert Tool """
    def __init__(self):
        global ini             
        self.inifile = ini
        self.ini = {}
        self.debug = False
        self.camera_count = 0
        self.cam = []        
        self.parser = configparser.ConfigParser()
        self.set_ini_config(self.inifile)        
               
    def set_ini_config(self, inifile):
        self.parser.read(inifile)
        
        for section in self.parser.sections():
            self.ini[section] = {}
            for option in self.parser.options(section):
                self.ini[section][option] = self.parser.get(section, option)
            if 'CAMERA' in section:
                self.cam.append(self.ini[section])


# -------------------------
# Real-time FER Application
# -------------------------
class FER_INT_ALG():

    def __init__(self):
        global ini
        ini = 'config.ini'
        self.Config = Config()
        self.trackingQueue = [Queue() for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count']))]        
        self.vaQueue = Queue()
        self.isReady = Value('i', 0)
        self.camTbl = {}

        global open_algorithm
        open_algorithm = True
        
        global key; global f
        global image_queue; global image_queue_move
        global cam_check
        
        global cname
        global image_batch
        global do_FER
        do_FER = False

        global faces
        faces = np.empty((200, 224, 224, 3))
        global net
        
        global valence, arousal
        valence, arousal = torch.zeros(1), torch.zeros(1)
        global fd_signal
        fd_signal = 0

        global cap_count
        cap_count = 0

        global face_Bbox
        face_Bbox = [(0,0), (0,0)]

        # Load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        modelPath = os.path.sep.join(["face_detector",
            "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        
        global encoder, regressor, disc
        
        encoder, regressor, _ = nn_output()
        encoder.load_state_dict(torch.load('weights/Encoder.t7'), strict=False)
        regressor.load_state_dict(torch.load('weights/FC_layer.t7'), strict=False)
        
        global emotion_list
        global emot_region
        emotion_list = ["angry", "sad", "happy", "pleased", "neutral"]
        emot_region = ""
        
        encoder.eval()
        regressor.eval()

        image_queue = []
        image_queue_move = []
        image_batch = []
        key = [0,0,0,0]; f = 0

        # We can set multiple cameras in the future!
        cname = {'1th_left'  : 0,}
#                 '1th_right' : 1,}

        cam_check = [0,0,0,0]
        
    def run(self):        
        camInfoList = []        
        camTbl = {}        
        global key

        for idx in range(0, int(self.Config.ini['COMMON']['camera_count'])):
            camInfo = {}
            camUUID = uuid.uuid4()
            
            camInfo.update({"uuid": camUUID})
            camInfo.update({"isready": self.isReady})
            camInfo.update({"tqueue": self.trackingQueue[idx]})
            camInfo.update({"vaqueue": self.vaQueue})            
            camInfo.update({"conf": self.Config.cam[idx]})
            camInfo.update({"globconf": self.Config})
            
            camInfoList.append(camInfo)
            camTbl.update({camUUID: camInfo})
        
        while (True):
            loop = asyncio.get_event_loop()

            loop.run_until_complete(async_handle_video_run(camInfoList))
            loop.close()
            if key == ord('q'):
                break
                

    def close(self):
        for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count'])):
            self.trackingQueue[idx].close()
            self.trackingQueue[idx].join_thread()
        self.vaQueue.close()
        self.vaQueue.join_thread()
        #self.VideoHdlrProc.stop()
        #self.AnalyzerHdlrProc.stop()
        #self.TrackingHdlrProc.stop()
        

if __name__ == "__main__":
    fer_int_alg = FER_INT_ALG()
    print ('Start... FER application')

    fer_int_alg.run()
    fer_int_alg.close()
    print("Completed FER application")

