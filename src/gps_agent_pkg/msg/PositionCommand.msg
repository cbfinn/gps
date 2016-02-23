int32 id  # ID must be echoed back in SampleResult
int8 mode  # control mode (i.e. joint space, task space), enum in gps_pb2
int8 arm  # which arm to execute on (actuator type enum in gps_pb2)
float64[] data
float64[] pd_gains
