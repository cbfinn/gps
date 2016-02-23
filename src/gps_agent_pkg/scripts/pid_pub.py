#!/usr/bin/env python
import rospy
import roslib; roslib.load_manifest('gps_agent_pkg')
from gps_agent_pkg.msg import PositionCommand
from gps_agent_pkg.msg import RelaxCommand

pos_pub = rospy.Publisher('/gps_controller_position_command', PositionCommand)
relax_pub = rospy.Publisher('/gps_controller_relax_command', RelaxCommand)
rospy.init_node('controller_pub_node')
r = rospy.Rate(10) # 10hz
count = 0
while not rospy.is_shutdown():
    if (count % 1000 < 250):
        new_msg = PositionCommand(mode=1, arm=1, data=[-0.75, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
        pos_pub.publish(new_msg)
    elif (count % 1000 < 500):
        new_msg = PositionCommand(mode=1, arm=1, data=[0.75, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
        pos_pub.publish(new_msg)
    else:
        new_msg_relax = RelaxCommand(arm=0)
        relax_pub.publish(new_msg_relax)
        new_msg_relax = RelaxCommand(arm=1)
        relax_pub.publish(new_msg_relax)

    count += 1
    r.sleep()
    #pub.publish("hello world")
