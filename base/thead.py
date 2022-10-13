# import threading, cv2

# from yolov5.utils.plots import Annotator, colors
# import tensorflow.keras as keras
# from vidgear.gears import CamGear
# from yolov5.utils.general import (xyxy2xywh)
# import torch, yolov5
# from yolov5.utils.torch_utils import select_device
# from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort
# class Video_Feed(threading.Thread):
#     def __init__(self, count, count_id):
#         self.count = count
#         self.count_id = count_id
#         threading.Thread.__init__(self)

#     def run(self, count, count_id):
#         def count_deep(box, w, h, id):
#                 # global count, count_id
#             center_circle = (int(box[0] + (box[2] - box[0])/2)) , (int(box[1] + (box[3] - box[1])/2))
#             if (int(box[1] + (box[3] - box[1])/2)) > (h - 50):
#                 if id not in count_id:
#                     count += 1
#                     count_id.append(id)
                        
#         model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

#         # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#         device = select_device('') # 0 for gpu, '' for cpu
#         # initialize deepsort
#         cfg = get_config()
#         cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
#         deepsort = DeepSort('osnet_x0_25',
#                     device,
#                     max_dist=cfg.DEEPSORT.MAX_DIST,
#                     max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
#                     )
#         names = model.module.names if hasattr(model, 'module') else model.names
#         print("bat dau maoooooooooo")
#         # url = "https://www.youtube.com/watch?v=DOrDtgdZzMo"
#         # time.sleep(2)
#         stream = CamGear(source="https://www.youtube.com/watch?v=UJr3gGBNbpo", stream_mode = True, logging=True).start()
#         # cap = cv2.VideoCapture("videos/Traffic.mp4")
#         model.conf = 0.45
#         model.iou = 0.5
#         model.classes = [0]
#         # model.classes = [0,64,39]
#         while True:
#             # ret, frame = cap.read()
#             frame = stream.read()
#             frame = cv2.resize(frame, (500, 250))
#             # frame = cv2.resize(frame, (500,250))
#             # if not ret:
#             #     print("Error: failed to capture image")
#             #     break
            
#             results = model(frame, augment=True)
#             # proccess
#             annotator = Annotator(frame, line_width=2, pil=not ascii) 
#             w, h = frame.shape[1], frame.shape[0]
#             color=(0,255,0)
#             start_point = (0, h-50)
#             end_point = (w, h-50)
#             cv2.line(frame, start_point, end_point, color, thickness=2)
#             thickness = 1
#             org = (50, 50)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             cv2.putText(frame, str(count), org, font, 
#                 fontScale, color, thickness, cv2.LINE_AA)
#             det = results.pred[0]
            
            
#             if det is not None and len(det):   
#                 xywhs = xyxy2xywh(det[:, 0:4])
                
                
#                 confs = det[:, 4]
#                 clss = det[:, 5]
#                 outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
                
#                 if len(outputs) > 0:
#                     for j, (output, conf) in enumerate(zip(outputs, confs)):
                        
#                         bboxes = output[0:4]
#                         id = output[4]
                        
#                         cls = output[5]
#                         count_deep(bboxes, w, h, id)
#                         c = int(cls)  # integer class
#                         label = f'{id} {names[c]} {conf:.2f}'
#                         annotator.box_label(bboxes, label, color=colors(c, True))
#             else:
#                 deepsort.increment_ages()

#             im0 = annotator.result()    
#             image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n') 

#             # if cv2.waitKey(1) & 0xFF == ord('q'):
#             #     break
            
#         # cap.release()
#         # cv2.destroyAllWindows()
