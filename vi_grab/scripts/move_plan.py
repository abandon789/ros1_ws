#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
#该代码是自己弄的，逆解求不出来，比较危险，但是运动范围比机械臂那边的多
from std_msgs.msg import String, Bool, Empty, Byte
import rospy, sys
from rm_msgs.msg import MoveJ_P, Arm_Current_State, ArmState, MoveL, MoveJ, Tool_Analog_Output, Stop
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from vi_msgs.msg import ObjectInfo
from geometry_msgs.msg import TransformStamped, PointStamped
from geometry_msgs.msg import Point, Quaternion
import threading
import math
# 全局变量
object_msg = String()
object_msg.data = ''
last_detected_object = None
terminal_input_thread = None
run_thread = True

# 工作空间限制（米）
work_limits= {
    'x_min': -0.7, 'x_max': 0.7,
    'y_min': -0.7, 'y_max': 0.7,
    'z_min': 0.2, 'z_max': 1.2
}
# 相机坐标系到joint7坐标系的旋转矩阵，通过手眼标定得到
rotation_matrix = np.array([
 [0.98748883, -0.02887373, 0.15502297],
 [0.01090677, 0.99324518, 0.1155208],
 [-0.15731133, -0.1123847, 0.98113344]
])

# 相机坐标系到机械臂joint7坐标系的平移向量，通过手眼标定得到
translation_vector = np.array([-0.05441651, -0.22274381, 3.17391599])

def work_ranges(x, y, z):
    """
    判断给定的目标点(x,y,z) 是否在预设的工作空间范围内。
    返回 True 表示在范围内，False 表示超出。
    """
    return (
        work_limits['x_min'] <= x <= work_limits['x_max'] and
        work_limits['y_min'] <= y <= work_limits['y_max'] and
        work_limits['z_min'] <= z <= work_limits['z_max']
    )

    

def convert(x, y, z, x1, y1, z1, rx, ry, rz):
    """
    函数功能：将相机坐标系下的物体坐标转换到机械臂基座坐标系
    输入参数：
        x, y, z: 相机坐标系下物体的坐标
        x1, y1, z1, rx, ry, rz: 机械臂末端的位姿（位置和欧拉角）
    返回值：物体在机械臂基座坐标系下的位姿
    """
    global rotation_matrix, translation_vector
    
    # 相机坐标系中的物体坐标
    obj_camera_coordinates = np.array([x, y, z])
    
    # 构建相机到末端(joint7)的变换矩阵
    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector
    
    # 构建末端(joint7)到工具的变换矩阵
    # joint7和工具坐标系之间的关系
    tool_pose = np.array([-0.0211, 0.01968, 0.1653, -3.125, 0.433, -2.723])
    tool_position = tool_pose[:3]
    tool_rotation = R.from_euler('xyz', tool_pose[3:], degrees=False).as_matrix()#求齐次矩阵
    
    T_end_effector_to_tool = np.eye(4)
    T_end_effector_to_tool[:3, :3] = tool_rotation
    T_end_effector_to_tool[:3, 3] = tool_position
    
    # 构建工具到基座的变换矩阵（从当前机械臂状态获取）
    # 使用机械臂当前位姿（x1, y1, z1, rx, ry, rz）构建变换矩阵
    current_position = np.array([x1, y1, z1])
    current_rotation = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()
    
    T_base_to_tool = np.eye(4)
    T_base_to_tool[:3, :3] = current_rotation
    T_base_to_tool[:3, 3] = current_position
    
    # 计算基座到末端（joint7）的变换矩阵
    T_base_to_end_effector = T_base_to_tool @ np.linalg.inv(T_end_effector_to_tool)#求逆变换
    
    # 将物体从相机坐标系转换到基座坐标系
    obj_camera_homo = np.append(obj_camera_coordinates, 1)  # 转为齐次坐标
    obj_end_effector_homo = T_camera_to_end_effector @ obj_camera_homo  # 相机坐标系 -> 末端坐标系
    obj_base_homo = T_base_to_end_effector @ obj_end_effector_homo  # 末端坐标系 -> 基座坐标系
    
    # 提取结果坐标
    obj_base_position = obj_base_homo[:3]
    
    # 保持原始目标的姿态（rx, ry, rz）
    obj_base_pose = np.array([
        obj_base_position[0], 
        obj_base_position[1], 
        obj_base_position[2],
        rx, ry, rz
    ])
    
    return obj_base_pose

def object_pose_callback(data):
    """
    函数功能：处理检测到的物体信息
    输入参数：ObjectInfo消息
    返回值：无
    """
    global object_msg, last_detected_object

    
    # 判断当前帧的识别结果是否有要抓取的物体
    if object_msg.data and data.object_class == object_msg.data:
        print(f"检测到目标: {data.object_class}")
        print(f"相机坐标系中的坐标: x={data.x:.4f}, y={data.y:.4f}, z={data.z:.4f}")
        
        # 保存检测到的目标信息
        last_detected_object = data
        
        # 等待当前的机械臂位姿
        try:
            arm_pose_msg = rospy.wait_for_message("/rm_driver/Arm_Current_State", Arm_Current_State, timeout=5)
            arm_orientation_msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=5)
            
            # 计算机械臂基坐标系下的物体坐标
            result = convert(
                data.x, data.y, data.z,
                arm_pose_msg.Pose[0], arm_pose_msg.Pose[1], arm_pose_msg.Pose[2],
                arm_pose_msg.Pose[3], arm_pose_msg.Pose[4], arm_pose_msg.Pose[5]
            )
            
            print(f"目标 '{data.object_class}' 在基座坐标系中的坐标: {result}")
            # 检查目标位置是否在工作空间内判断相机获取的距离机械臂能不能到
            if work_ranges(data.x, data.y, data.z):
                # 执行抓取动作
                catch(result, arm_orientation_msg)
                string_view()
                print("请输入命令：")
            else:
                print(f"警告: 目标位置 [{data.x:.4f}, {data.y:.4f}, {data.z:.4f}] 超出工作空间范围!")
                print(f"工作空间范围: X[{work_limits['x_min']}~{work_limits['x_max']}], " + 
                      f"Y[{work_limits['y_min']}~{work_limits['y_max']}], " + 
                      f"Z[{work_limits['z_min']}~{work_limits['z_max']}]")
                rospy.logwarn("目标位置超出工作空间安全范围，已取消动作，请重新输入命令")
                string_view()
            
            # 清除object_msg的信息，之后二次发布抓取物体信息可以再执行
            object_msg.data = ''
            
        except rospy.ROSException as e:
            rospy.logerr(f"获取机械臂位姿超时: {e}")
        except Exception as e:
            rospy.logerr(f"处理目标位置时出错: {e}")

def movej_type(joint, speed):
    '''
    函数功能：通过输入机械臂每个关节的数值（弧度），让机械臂以指定速度运动到指定姿态
    输入参数：[joint1,joint2,joint3,joint4,joint5,joint6,joint7]、speed
    返回值：无
    '''
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
    rospy.sleep(1)
    move_joint = MoveJ()
    move_joint.joint = joint
    move_joint.speed = speed
    moveJ_pub.publish(move_joint)
    print(f"已发送关节空间运动指令: joints={joint}, speed={speed}")

def movejp_type(pose, speed):
    '''
    函数功能：通过输入机械臂末端的位姿数值，让机械臂以指定速度运动到指定位姿
    输入参数：pose（position.x...z, orientation.x...w）、speed
    返回值：无
    '''
    moveJ_P_pub = rospy.Publisher("/rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=1)
    rospy.sleep(1)
    move_joint_pose = MoveJ_P()
    move_joint_pose.Pose.position.x = pose[0]
    move_joint_pose.Pose.position.y = pose[1]
    move_joint_pose.Pose.position.z = pose[2]
    move_joint_pose.Pose.orientation.x = pose[3]
    move_joint_pose.Pose.orientation.y = pose[4]
    move_joint_pose.Pose.orientation.z = pose[5]
    move_joint_pose.Pose.orientation.w = pose[6]
    move_joint_pose.speed = speed
    moveJ_P_pub.publish(move_joint_pose)
    print(f"已发送位姿空间运动指令: pose={pose}, speed={speed}")

def movel_type(pose, speed):
    '''
    函数功能：通过输入机械臂末端的位姿数值，让机械臂以指定速度直线运动到指定位姿
    输入参数：pose（position.x...z, orientation.x...w）、speed
    返回值：无
    '''
    moveL_pub = rospy.Publisher("/rm_driver/MoveL_Cmd", MoveL, queue_size=1)
    rospy.sleep(1)
    move_line_pose = MoveL()
    move_line_pose.Pose.position.x = pose[0]
    move_line_pose.Pose.position.y = pose[1]
    move_line_pose.Pose.position.z = pose[2]
    move_line_pose.Pose.orientation.x = pose[3]
    move_line_pose.Pose.orientation.y = pose[4]
    move_line_pose.Pose.orientation.z = pose[5]
    move_line_pose.Pose.orientation.w = pose[6]
    move_line_pose.speed = speed
    moveL_pub.publish(move_line_pose)
    print(f"已发送直线运动指令: pose={pose}, speed={speed}")

def arm_ready_pose():
    '''
    函数功能：运动到识别姿态
    输入参数：无
    返回值：无
    '''
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
    rospy.sleep(1)
    pic_joint = MoveJ()
    '''
    需要启动我已经集成好的moveit启动文件roslaunch rm_bringup gen72_robot.launch 
    通过教/moveit运动到合适的位置
    角度值通过roslaunch get_arm_state get_arm_state_demo.launch 获取机械臂角度信息
    复制的信息为joint angle state is里面的内容到get_view_angle里面就是机械臂一开始观察目标点的位姿
    '''
    get_view_angle = [-27.582001, 26.969999, 1.900000, 34.962002, 26.143000, -7.594000, -31.268999]#后面需要自己换位姿关系就改这里
    pic_joint.joint = [i*math.pi/180 for i in get_view_angle]  # 转换为弧度制
    pic_joint.speed = 0.1  # 速度
    moveJ_pub.publish(pic_joint)  # 发布movej运动位置信息
    print("移动到准备位姿...")

def complete_pose():
    '''
    函数功能：运动到初始姿态
    输入参数：无
    返回值：无
    '''
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
    rospy.sleep(1)
    pic_joint = MoveJ()
    '''
    需要启动我已经集成好的moveit启动文件roslaunch rm_bringup gen72_robot.launch 
    通过教/moveit运动到合适的位置
    角度值通过roslaunch get_arm_state get_arm_state_demo.launch 获取机械臂角度信息
    复制的信息为joint angle state is里面的内容到get_view_angle里面就是机械臂一开始观察目标点的位姿
    '''
    home_angle = [3.212000, 1.241000, 3.098000, -147.791000, -0.109000, 86.931000, -31.260000]#后面需要自己换位姿关系就改这里
    pic_joint.joint = [i*math.pi/180 for i in home_angle]  # 转换为弧度制
    pic_joint.speed = 0.1
    moveJ_pub.publish(pic_joint)
    print("移动到初始位姿...")

def catch(result, arm_orientation_msg):
    '''
    函数功能：机械臂执行抓取动作
    输入参数：经过convert函数转换得到的'result'和机械臂当前的四元数位姿'arm_orientation_msg'
    返回值：无
    '''
    # 将姿态从欧拉角转换为四元数
    orientation_q = arm_orientation_msg.Pose.orientation
    
    print("=== 开始执行抓取序列 ===")
    
    # 第一步：移动到目标点上方
    approach_pose = [
        result[0]-0.05,  # 离目标点还有距离
        result[1],
        result[2]+0.02,   # 抬高
        orientation_q.x,
        orientation_q.y,
        orientation_q.z,
        orientation_q.w
    ]
    print(f"步骤1: 移动到目标上方 - {approach_pose}")
    movejp_type(approach_pose, 0.1)#位姿+速度，测试的时候先给0.01看机械臂大致运动趋势，对了在给高一点
    rospy.sleep(15)  #延时
    
    # 第二步：降低高度靠近目标
    approach_pose[2] = result[2] + 0.02  # 降低到接近目标
    print(f"步骤2: 降低到目标附近 - {approach_pose}")
    movel_type(approach_pose, 0.05)
    rospy.sleep(15) 

    # 第二步：直线移动到目标位置
    target_pose = [
        result[0],
        result[1],
        result[2],
        orientation_q.x,
        orientation_q.y,
        orientation_q.z,
        orientation_q.w
    ]
    print(f"步骤2: 直线移动到目标位置 - {target_pose}")
    movel_type(target_pose, 0.05)
    rospy.sleep(3)  # 等待运动完成
    
    # 第四步：按下电梯按钮
    print(f"步骤4: 暂停1秒，按电梯")
    rospy.sleep(1)
    
    # 第五步：后退0.05
    print(f"步骤5: 返回到目标上方 - {approach_pose}")
    movel_type(approach_pose, 0.05)
    rospy.sleep(3)  # 等待运动完成
    
    # 第六步：返回到准备姿态
    print(f"步骤6: 返回到准备姿态")
    arm_ready_pose()
    
    print("=== 按电梯执行完成 ===")

def set_tool_voltage(voltage=24):
    '''
    函数功能：设置工具端电压输出
    输入参数：电压值（默认24V）
    返回值：无
    '''
    pub_tool_voltage = rospy.Publisher("/rm_driver/Tool_Analog_Output", Tool_Analog_Output, queue_size=1)
    rospy.sleep(1)
    set_vol = Tool_Analog_Output()
    set_vol.voltage = voltage
    pub_tool_voltage.publish(set_vol)
    print(f"已设置工具端电压: {voltage}V")

def emergency_stop():
    '''
    函数功能：紧急停止
    输入参数：无
    返回值：无
    '''
    stop_pub = rospy.Publisher("/rm_driver/Emergency_Stop", Empty, queue_size=1)
    rospy.sleep(0.5)  # 等待连接
    
    stop_pub.publish(Empty())
    rospy.loginfo("已发送紧急停止信号")

def clear_error():
    '''
    函数功能：清除关节错误
    输入参数：无
    返回值：无
    '''
    clear_pub = rospy.Publisher("/rm_driver/Clear_System_Err", Empty, queue_size=1)
    clear_pub.publish(Empty())
    print("已清除系统错误")

def set_power(state):
    '''
    函数功能：控制机械臂上电/断电
    输入参数：state (1=上电, 0=断电)
    返回值：无
    '''
    power_pub = rospy.Publisher("/rm_driver/SetArmPower", Byte, queue_size=4)
    power_msg = Byte()
    power_msg.data = state
    power_pub.publish(power_msg)
    print(f"机械臂电源状态已设置为: {'上电' if state == 1 else '断电'}")

def change_tool_frame(tool_name="hand_frame"):
    '''
    函数功能：切换工具坐标系
    输入参数：工具名称
    返回值：无
    '''
    from rm_msgs.msg import ChangeTool_Name
    tool_pub = rospy.Publisher("/rm_driver/ChangeToolName_Cmd", ChangeTool_Name, queue_size=4)
    tool_msg = ChangeTool_Name()
    tool_msg.toolname = tool_name
    tool_pub.publish(tool_msg)
    print(f"已切换到工具坐标系: {tool_name}")
def string_view():
    print("\n===== 机械臂控制终端 =====")
    print("命令列表:")
    print("  数字 1-12, 'up', 'down', 'LCD', 'open', 'close', 'warning': 选择目标对象")
    print("  h: 返回到初始姿态")
    print("  g: 返回到准备姿态")
    print("  s: 紧急停止和断电")
    print("  q: 机械臂断电")
    print("  r: 机械臂上电")
    print("  d: 清除关节错误")
    print("  t: 切换工具坐标系")
    print("  x或 Ctrl+C: 退出程序")
    print("请输入'a'回到初始位置并断电")
    print("==========================\n")
    

def terminal_input():
    '''
    函数功能：处理终端输入
    输入参数：无
    返回值：无
    '''
    global object_msg, run_thread
    string_view()

    
    while run_thread:
        try:
            cmd = input("请输入命令: ")
            string_view()
            
            if cmd.lower() in ['x', 'exit', 'quit']:
                print("退出程序中...")
                run_thread = False
                rospy.signal_shutdown("用户退出")
                break
            elif cmd.lower() in ['a']:
                complete_pose()
                rospy.sleep(20)
                set_power(0)

            elif cmd.lower() in ['h']:
                print("返回到初始姿态中...")
                complete_pose()
                
            elif cmd.lower() in ['s']:
                print("紧急停止...")
                emergency_stop()
                set_power(0)#由于发送关节规划话题消息一直在发布，断电避免意外
                
            elif cmd.lower() in ['q']:
                print("机械臂断电...")
                set_power(0)
                
            elif cmd.lower() in ['r']:
                print("机械臂上电...")
                set_power(1)
                
            elif cmd.lower() in ['d']:
                print("清除关节错误...")
                clear_error()
            
            elif cmd.lower() in ['g']:
                print("回到准备姿态中...")
                arm_ready_pose()
                
            elif cmd.lower() in ['t']:
                tool_name = input("请输入工具坐标系名称 [默认:hand_frame]: ") or "初始工具坐标系Arm_Tip"
                change_tool_frame(tool_name)
                
            elif cmd in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 
                         'up', 'down', 'LCD', 'open', 'close', 'warning']:
                print(f"选择目标类别: {cmd}")
                object_msg.data = cmd
                
                # 如果最后检测到的目标与当前选择匹配，直接处理
                if last_detected_object and last_detected_object.object_class == cmd:
                    print("使用最近检测到的目标位置...")
                    object_pose_callback(last_detected_object)
                else:
                    print("等待检测到目标...")
            else:
                print(f"未知命令: {cmd}")
                
        except Exception as e:
            print(f"命令处理错误: {e}")

def main():
    global terminal_input_thread, run_thread
    rospy.init_node('object_catch')
    # 发布机械臂状态
    pub_arm_pose = rospy.Publisher("/rm_driver/GetCurrentArmState", Empty, queue_size=1)
    
    # 订阅目标物体位置信息
    sub_object_pose = rospy.Subscriber("/object_pose", ObjectInfo, object_pose_callback, queue_size=1)
    
    # 设置初始状态
    try:
        print("初始化机械臂...")

        # 确保机械臂上电
        set_power(1)
        rospy.sleep(5)#稍微久一点避免后续操作发不出去
        
        # 清除系统错误
        clear_error()
        rospy.sleep(3)

        # 切换工具坐标系
        change_tool_frame("hand_frame")
        rospy.sleep(4)



        
        # 移动到准备姿态
        print("移动到准备姿态...")
        arm_ready_pose()
        
    except Exception as e:
        rospy.logerr(f"初始化错误: {e}")
    
    # 创建终端输入线程
    terminal_input_thread = threading.Thread(target=terminal_input)
    terminal_input_thread.daemon = True
    terminal_input_thread.start()
    
    # 保持节点运行
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        run_thread = False
        if terminal_input_thread:
            terminal_input_thread.join(timeout=1.0)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass