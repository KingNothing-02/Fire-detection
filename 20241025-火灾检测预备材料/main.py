import time
import threading
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from general import Profile
from QTtest3 import Ui_MainWindow
import sys
import cv2
from ultralytics import YOLO


class MyWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)  # 初始化UI界面
        self.setupCamera(0)
        self.setupWidget()
        self.slot_connect()  # 信号槽函数

    def setupCamera(self, URL):
        """初始化yolov8参数与模型"""
        self.weights = 'F:/PyCharm Community Edition 2024.2/deeplearning1/runs/detect/train/weights/best.pt'
        self.conf_thres = 0.25  # 置信度阈值，识别后会输出一个数字代表程序认为物体是某个类别的可能性。如果可能性小于阈值则认为识别失败，不输出结果
        self.iou_thres = 0.45  # 用于指定非极大值抑制（NMS）的IoU阈值 当两个识别框有重合时，计算一个数值：两个识别框的重叠面积/两个识别框的面积和，若数值大于某个阈值，则认为是同一物体，删除另一个识别框。若NMS不开启，则不执行以上步骤
        self.max_det = 1000  # 指定每张图像的最大检测次数
        self.agnostic_nms = False  # 是否开启NMS
        self.half = False  # 是否使用半精度浮点数进行推理，用英伟达的gpu推理时才能用，可以减少计算量，加快推理速度。你的显卡是AMD的，只能用cpu推理
        self.augment = False  # 是否增强图像后处理。如果原来的图片的物体的特征不明显，可以对图像增强后再识别，增加识别的准确率，时间成本也增加了
        self.model = YOLO(self.weights)  # 加载模型
        """初始化摄像头参数"""
        self.dt = Profile()
        self.video_dt = Profile()
        self.camera_URL = URL
        self.cap = cv2.VideoCapture()
        self.img_distinguish = None
        self.image = None
        self.open_close_flag = False  # 是否打开摄像头
        self.distinguish_flag = False
        self.image_path = None
        self.video_path = None
        self.img_save_num = 1
        self.video_save_num = 1
        self.result_num = 1

    def setupWidget(self):
        self.label.setScaledContents(True)
        self.label_2.setScaledContents(True)
        self.textBrowser.append("这里输出结果\n")
        self.widget_conf.sl.setMaximum(100)
        self.widget_conf.sl.setMinimum(0)
        self.widget_iou.sl_2.setMaximum(100)
        self.widget_iou.sl_2.setMinimum(0)
        self.widget_max_det.sl_3.setMaximum(5000)
        self.widget_max_det.sl_3.setMinimum(0)
        self.widget_conf.sl.setValue(25)
        self.widget_iou.sl_2.setValue(50)
        self.widget_max_det.sl_3.setValue(2500)
        self.radioButton_yolo_half_false.setChecked(True)
        self.radioButton_yolo_nms_false.setChecked(True)
        self.radioButton_yolo_line_2.setChecked(True)
        self.label.setText("原样本")
        self.label_2.setText("结果")

    def pushButton_open_close_camera_cb(self):
        if not self.open_close_flag:
            thr = threading.Thread(target=self.open_camera)
            thr.start()
        else:
            if self.distinguish_flag:
                QMessageBox.warning(self, "warning", "请先结束识别", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.open_close_flag = False
                time.sleep(0.05)
                self.pushButton_open_close_camera.setText("打开摄像头")
                self.label_fps.setText("0")
                self.label.clear()
                self.label_2.clear()
                self.cap.release()
    def open_camera(self):
        flag = self.cap.open(self.camera_URL)  # 启用网络摄像头
        if not flag:
            QMessageBox.warning(self, "warning", "摄像头打开失败！", buttons=QtWidgets.QMessageBox.Ok)
        else:
            self.open_close_flag = True
            self.pushButton_open_close_camera.setText("关闭摄像头")
            thr = threading.Thread(target=self.keep_taking_photo)
            thr.start()
    def keep_taking_photo(self):
        while self.open_close_flag:
            with self.dt:
                ret, image = self.cap.read()
                if ret:
                    self.image = image
                    self.show_photo_process(self.image)
            if self.dt.dt > 0.01:
                self.label_fps.setText(str(round(1 / self.dt.dt, 1)))

    def show_photo_process(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(img))
        if self.distinguish_flag:
            image_distinguish = self.model.predict(
                                source=image,
                                conf=self.conf_thres,
                                iou=self.iou_thres,
                                max_det=self.max_det,
                                half=self.half,
                                agnostic_nms=self.agnostic_nms,
                                augment=self.augment,
            )[0].plot()
        else:
            image_distinguish = image
        image_distinguish = cv2.cvtColor(image_distinguish, cv2.COLOR_BGR2RGB)
        image_distinguish = QtGui.QImage(image_distinguish.data, image_distinguish.shape[1], image_distinguish.shape[0], QtGui.QImage.Format_RGB888)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(image_distinguish))

    def pushButton_open_close_distinguish_cb(self):  # 识别
        if self.open_close_flag:
            if not self.distinguish_flag:
                self.distinguish_flag = True
                self.pushButton_open_close_distinguish.setText("关闭识别")
            else:
                self.distinguish_flag = False
                self.pushButton_open_close_distinguish.setText("开启识别")
        else:
            QMessageBox.warning(self, 'warning', "未打开摄像头", buttons=QtWidgets.QMessageBox.Ok)

    def yolo_setting_update(self):
        self.conf_thres = self.widget_conf.sl.value() / 100
        self.iou_thres = self.widget_iou.sl.value() / 100
        self.max_det = int(self.widget_max_det.sl.value() / 100)
        self.half = self.radioButton_yolo_half_true.isChecked()
        self.agnostic_nms = self.radioButton_yolo_nms_true.isChecked()
        self.augment = self.radioButton_yolo_line_1.isChecked()

    "-------------------------------------------------图片与视频检测------------------------------------------------------"

    def pushButton_select_photo_cb(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QtGui.QPixmap(self.image_path)
            self.label.setPixmap(pixmap)

    def pushButton_img_distinguish_cb(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            image = self.model.predict(
                    source=image,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    max_det=self.max_det,
                    half=self.half,
                    agnostic_nms=self.agnostic_nms,
                    augment=self.augment,
            )
            print(image)
            img = image[0].plot()
            cv2.imwrite(f"distinguish_imgs/photo{self.img_save_num}.jpg", img)
            self.textBrowser.append(f"检测结果{self.result_num}:\n目标数量:{image[0].verbose()}\n文件保存路径:distinguish_imgs/photo{self.img_save_num}.jpg\n")
            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
            pixmap = QtGui.QPixmap(f"distinguish_imgs/photo{self.img_save_num}.jpg")
            print(3)
            self.label_2.setPixmap(pixmap)
            self.img_save_num += 1
            self.result_num += 1
            self.image_path = None
        else:
            self.textBrowser.append("未导入图片！\n")
            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)

    def slot_connect(self):
        self.timer_widget = QtCore.QTimer()
        self.timer_widget.timeout.connect(self.yolo_setting_update)
        self.timer_widget.start(50)
        self.pushButton_open_close_camera.clicked.connect(self.pushButton_open_close_camera_cb)
        self.pushButton_open_close_distinguish.clicked.connect(self.pushButton_open_close_distinguish_cb)
        self.pushButton_select_photo.clicked.connect(self.pushButton_select_photo_cb)
        self.pushButton_select_video.clicked.connect(self.pushButton_select_video_cb)
        self.pushButton_img_distinguish.clicked.connect(self.pushButton_img_distinguish_cb)
        self.pushButton_video_distinguish.clicked.connect(self.pushButton_video_distinguish_cb)

    def pushButton_select_video_cb(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.video_path = file_path
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def pushButton_video_distinguish_cb(self):
        thr = threading.Thread(target=self.video_process)
        thr.start()

    def video_process(self):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, 'warning', "视频导入失败", buttons=QtWidgets.QMessageBox.Ok)
                return
            else:
                with self.video_dt:
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    videowriter = cv2.VideoWriter(f"./distinguish_videos/video{self.video_save_num}.mp4",
                                                       cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                    while True:
                        start_time = time.time()
                        ret, frame = cap.read()
                        if not ret:
                            break
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
                        self.label.setPixmap(QtGui.QPixmap.fromImage(image))
                        frame = self.model.predict(
                                source=frame,
                                conf=self.conf_thres,
                                iou=self.iou_thres,
                                max_det=self.max_det,
                                half=self.half,
                                agnostic_nms=self.agnostic_nms,
                                augment=self.augment,
                        )[0].plot()
                        videowriter.write(frame)
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
                        self.label_2.setPixmap(QtGui.QPixmap.fromImage(image))
                        end_time = time.time()
                        if (end_time - start_time) * 2 < 1 / 25:
                            time.sleep(1 / 25 - (end_time - start_time) * 2)
                    videowriter.release()
                    self.video_path = None
                    self.video_save_num += 1
                self.textBrowser.append(f"检测结果{self.result_num}:\n运行时间:{self.video_dt.dt:.1f}s\n文件保存路径:distinguish_videos/video{self.video_save_num}.mp4\n")
                self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                self.result_num += 1
        else:
            self.textBrowser.append(f"未导入视频!\n")
            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
