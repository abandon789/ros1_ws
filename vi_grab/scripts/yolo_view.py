#!/usr/bin/env python3
# coding=utf-8
import os
import rospy
import numpy as np
import cv2
import math
import threading
from ultralytics import YOLO
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from vi_msgs.msg import ObjectInfo  # 自定义ROS msg

# 相机内参
fx = 1033.4110107421875
fy = 1033.2586669921875
cx = 963.51202392578125
cy = 541.18023681640625


bridge = CvBridge()

# 共享数据锁和最新数据
data_lock = threading.Lock()
latest_data = None

# ROS 回调函数：存储最新的彩色图像和深度图像
def callback(color_msg, depth_msg):
    global latest_data
    with data_lock:
        latest_data = (color_msg, depth_msg)

# 获取对应像素点的3D坐标
def get_3d_camera_coordinate(u, v, depth_value):
    """
    计算像素点的3D坐标
    u, v: 像素坐标
    depth_value: 深度值（米）
    返回：相机坐标系下的3D坐标 (x, y, z)
    """
    # 相机内参
    z = depth_value*0.001# 深度值，转换为米，和机械臂位姿单位对应,厂家说不同的相机型号获取的深度单位不一样，需要和实际的测量在再转换
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)

def main():
    global latest_data  # 声明使用全局变量
    # 加载YOLO模型
    yolo_path= os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(yolo_path, "yolo_v8_model","best.pt")#模型路径,这个路径用于复用的
    # model_path = '/home/lzh/opencv_ws/src/cv_pkg/scripts/best.pt'  #模型路径
    try:
        model = YOLO(model_path) 
    except:
        rospy.logerr("无法加载模型，请检查模型路径: %s", model_path)
        return
    rospy.loginfo("完成YoloV8模型加载")
    rospy.init_node("object_detect", anonymous=True)

    # 定义话题发布的名字，消息类别，消息列表大小
    object_pub = rospy.Publisher("object_pose", ObjectInfo, queue_size=10)
    
    # 订阅彩色图像和深度图像话题
    color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
    
    # 同步深度相机和rgb相机数据
    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
    ts.registerCallback(callback)
    
    object_info_msg = ObjectInfo()
    
    rate = rospy.Rate(30)  # 帧数
    
    while not rospy.is_shutdown():
        with data_lock:
            if latest_data is not None:
                color_msg, depth_msg = latest_data
                latest_data = None
            else:
                rate.sleep()
                continue
        
        try:
            # 将ROS消息转换为OpenCV格式
            color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
            if color_image.shape[:2] != depth_image.shape[:2]:
                rospy.logerr("彩色图像和深度图像尺寸不匹配: %s vs %s", color_image.shape, depth_image.shape)
                continue
            
            # 使用 YOLOv8 进行目标检测
            results = model.predict(color_image, conf=0.7)#置信度高于70%才用，避免检测到不合适的，后续如果识别精度不高可以调到80%左右
            detected_boxes = results[0].boxes.xyxy  # 获取边界框坐标
            data = results[0].boxes.data.cpu().tolist()
            canvas = results[0].plot()
            
            # 处理每个检测到的目标 
            for i, (row, box) in enumerate(zip(data, detected_boxes)):
                id = int(row[5])
                name = results[0].names[id]
                x1, y1, x2, y2 = map(int, box)  # 获取边界框坐标
                
                # 计算边界框中心点
                ux = int((x1 + x2) / 2)
                uy = int((y1 + y2) / 2)
                # 在ROS接口中，从depth_image获取深度值
                depth_value = depth_image[uy, ux]
                
                # 检查深度值是否有效
                if np.isfinite(depth_value) and depth_value > 0:
                    camera_coordinate = get_3d_camera_coordinate(ux, uy, depth_value)
                    
                    # 格式化坐标显示
                    formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"
                    
                    # 在图像上显示中心点和坐标
                    cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                    cv2.putText(canvas, str(formatted_camera_coordinate), (ux + 20, uy + 10), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    
                    # 发布目标位置信息
                    object_info_msg.object_class = str(name)
                    object_info_msg.x = float(camera_coordinate[0])
                    object_info_msg.y = float(camera_coordinate[1])
                    object_info_msg.z = float(camera_coordinate[2])
                    rospy.loginfo(object_info_msg)
                    object_pub.publish(object_info_msg)
                else:
                    rospy.logwarn("检测点 (%d, %d) 周围无有效深度值", ux, uy)
            
            # 显示检测结果
            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('detection', canvas)
            key = cv2.waitKey(1)
            
            # 按下 esc 或者 'q' 退出程序和图像界面
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
                
        except CvBridgeError as e:
            rospy.logerr("CvBridge 错误: %s", e)
        except Exception as e:
            rospy.logerr("处理图像时发生异常: %s", e)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass