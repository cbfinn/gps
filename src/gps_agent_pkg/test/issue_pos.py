import rospy
import roslib
roslib.load_manifest('gps_agent_pkg')
import gps_agent_pkg
from gps_agent_pkg.msg import PositionCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
import numpy as np
from gps.proto.gps_pb2 import *

POS_COM_TOPIC = '/gps_controller_position_command'
TRIAL_COM_TOPIC = '/gps_controller_trial_command'
TEST_TOPIC = '/test_sub'

def listen(msg):
    print msg.__class__

def main():
    rospy.init_node('issue_com')
    pub = rospy.Publisher(POS_COM_TOPIC, PositionCommand, queue_size=10)
    test_pub = rospy.Publisher(TEST_TOPIC, Empty, queue_size=10)
    sub = rospy.Subscriber(POS_COM_TOPIC, PositionCommand, listen)
    #sub = rospy.Subscriber('/joint_states', JointState, listen)

    pc = PositionCommand()
    pc.mode = JOINT_SPACE
    #pc.arm = PositionCommand.LEFT_ARM
    pc.arm = 1#PositionCommand.RIGHT_ARM
    pc.data = np.zeros(7)

    r = rospy.Rate(1)
    #while not rospy.is_shutdown():
    #    pub.publish(pc)
    #    r.sleep()
    #    print 'published!'
    r.sleep()
    test_pub.publish(Empty())
    pub.publish(pc)

print "Testing"
if __name__ == "__main__":
    main()
