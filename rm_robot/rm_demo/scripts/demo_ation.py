#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#如果python3运行不了就改为python
#模拟带爪避障
import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.msg import PlanningScene, ObjectColor, AttachedCollisionObject, CollisionObject
from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive

class MoveItObstaclesDemo:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        
        # 初始化ROS节点
        rospy.init_node('moveit_obstacles_demo')
        
        # 初始化场景对象
        scene = PlanningSceneInterface()
        
        # 创建一个发布场景变化信息的发布者
        self.scene_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=5)
        
        # 创建一个存储物体颜色的字典对象
        self.colors = dict()
        
        # 等待场景准备就绪
        rospy.sleep(1)
                        
        # 初始化需要使用move group控制的机械臂中的arm group
        arm = MoveGroupCommander('arm')
        
        # 获取终端link的名称
        end_effector_link = arm.get_end_effector_link()
        rospy.loginfo("End effector link: %s", end_effector_link)
        
        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        arm.set_goal_position_tolerance(0.01)
        arm.set_goal_orientation_tolerance(0.05)
       
        # 当运动规划失败后，允许重新规划
        arm.allow_replanning(True)
        
        # 设置目标位置所使用的参考坐标系
        reference_frame = 'base_link'
        arm.set_pose_reference_frame(reference_frame)
        
        # 设置每次运动规划的时间限制：5s
        arm.set_planning_time(5)
        
        # 设置场景物体的名称
        table_id = 'table'
        box1_id = 'box1'
        box2_id = 'box2'
        gripper_id = 'gripper'
        
        # 移除场景中之前运行残留的物体
        scene.remove_world_object(table_id)
        scene.remove_world_object(box1_id)
        scene.remove_world_object(box2_id)
        
        # 尝试移除之前可能存在的机械爪对象
        scene.remove_attached_object(end_effector_link, gripper_id)
        scene.remove_world_object(gripper_id)
        
        rospy.sleep(1)
        rospy.loginfo("Scene cleared of previous objects")
        
        # 设置桌面的高度
        table_ground = 0.35

        # 设置table、box1和box2的三维尺寸
        table_size = [0.2, 0.8, 0.03]
        box1_size = [0.1, 0.05, 0.1]
        box2_size = [0.05, 0.05, 0.3]
        
        # 设置机械爪的尺寸 (高17cm，半径为5cm的圆柱体)
        # 为避免初始状态碰撞，将尺寸稍微缩小
        gripper_height = 0.17  # 高
        gripper_radius = 0.05  # 半径
        
        # 在此处立即添加机械爪对象
        try:
            # 创建机械爪对象
            gripper_pose = PoseStamped()
            gripper_pose.header.frame_id = end_effector_link
            gripper_pose.pose.position.z = gripper_height / 2.0
            gripper_pose.pose.orientation.w = 1.0
            
            scene.add_cylinder(gripper_id, gripper_pose, gripper_height, gripper_radius)
            rospy.loginfo("夹爪已经添加")
            rospy.sleep(0.5)
            
            # 将机械爪附加到末端执行器
            touch_links = [end_effector_link]  
            # 机械爪与末端执行器接触
            scene.attach_object(gripper_id, end_effector_link, touch_links=touch_links)
            rospy.loginfo("夹爪已经添加到机械臂上")
            rospy.sleep(0.5)
            
            # 设置机械爪颜色
            self.setColor(gripper_id, 0, 0, 0.8, 0.6)
            self.sendColors()
        except Exception as e:
            rospy.logerr("Error adding gripper: %s", str(e))

        # 清空可能存在的所有路径约束
        arm.clear_path_constraints()
        
        # 先确保机械臂处于安全位置，再添加场景物体
        rospy.loginfo("Moving to zero position before adding obstacles")
        arm.set_named_target('zero')
        success_zero = arm.go(wait=True)
        rospy.loginfo("Move to zero position: %s", "Success" if success_zero else "Failed")
        rospy.sleep(1)
        
        # 添加桌子
        table_pose = PoseStamped()
        table_pose.header.frame_id = reference_frame
        table_pose.pose.position.x = 0.33
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = 0.65
        table_pose.pose.orientation.w = 1.0
        scene.add_box(table_id, table_pose, table_size)
        rospy.loginfo("Table added to scene")
        rospy.sleep(0.5)
        
        # 将桌子设置成红色
        self.setColor(table_id, 0.8, 0, 0, 1.0)
        self.sendColors()
        
        # 先移动到home位置
        rospy.loginfo("Moving to home position")
        arm.set_named_target('home')
        success_home = arm.go(wait=True)
        rospy.loginfo("Move to home: %s", "Success" if success_home else "Failed")
        rospy.sleep(1)
        
        # 获取当前末端执行器位置
        current_pose = arm.get_current_pose(end_effector_link)
        rospy.loginfo("Current end effector position: x=%s, y=%s, z=%s", 
                     current_pose.pose.position.x, 
                     current_pose.pose.position.y, 
                     current_pose.pose.position.z)
        
        # 尝试执行一些简单关节移动
        try:
            # 获取当前关节角度
            current_joints = arm.get_current_joint_values()
            if len(current_joints) > 0:
                rospy.loginfo("Current first joint value: %s", current_joints[0])
                
                # 小幅度修改第一个关节
                test_joints = list(current_joints)
                test_joints[0] += 0.1  # 第一个关节旋转0.1弧度
                
                rospy.loginfo("Testing joint movement")
                arm.set_joint_value_target(test_joints)
                joint_success = arm.go(wait=True)
                rospy.loginfo("Joint movement test: %s", "Success" if joint_success else "Failed")
                rospy.sleep(1)
        except Exception as e:
            rospy.logerr("Error during joint movement: %s", str(e))
        
        # 尝试移动到预定位置
        try:
            rospy.loginfo("Moving back to forward position")
            arm.clear_pose_targets()
            arm.clear_path_constraints()
            arm.set_named_target('forward')
            success_final = arm.go(wait=True)
            rospy.loginfo("Final move to zero: %s", "Success" if success_final else "Failed")
        except Exception as e:
            rospy.logerr("Error during final movement: %s", str(e))
        
        # 最后回到zero位置
        try:
            rospy.loginfo("Moving back to zero position")
            arm.clear_pose_targets()
            arm.clear_path_constraints()
            arm.set_named_target('zero')
            success_final = arm.go(wait=True)
            rospy.loginfo("Final move to zero: %s", "Success" if success_final else "Failed")
        except Exception as e:
            rospy.logerr("Error during final movement: %s", str(e))
        
        # 回到初始位置
        try:
            rospy.loginfo("Moving back to home position")
            arm.clear_pose_targets()
            arm.clear_path_constraints()
            arm.set_named_target('home')
            success_final = arm.go(wait=True)
            rospy.loginfo("Final move to zero: %s", "Success" if success_final else "Failed")
        except Exception as e:
            rospy.logerr("Error during final movement: %s", str(e))
        
        # 关闭并退出moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
        
    # 设置场景物体的颜色
    def setColor(self, name, r, g, b, a = 0.9):
        # 初始化moveit颜色对象
        color = ObjectColor()
        
        # 设置颜色值
        color.id = name          
        color.color.r = r
        color.color.g = g
        color.color.b = b
        color.color.a = a
        
        # 更新颜色字典
        self.colors[name] = color

    # 将颜色设置发送并应用到moveit场景当中
    def sendColors(self):
        # 初始化规划场景对象
        p = PlanningScene()

        # 需要设置规划场景是否有差异     
        p.is_diff = True
        
        # 从颜色字典中取出颜色设置
        for color in self.colors.values():
            p.object_colors.append(color)
        
        # 发布场景物体颜色设置
        self.scene_pub.publish(p)

if __name__ == "__main__":
    try:
        MoveItObstaclesDemo()
    except KeyboardInterrupt:
        raise