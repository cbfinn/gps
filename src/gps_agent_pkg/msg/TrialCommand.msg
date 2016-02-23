# This message is published to the C++ controller to start
# a trial
int32 id  # ID must be echoed back in SampleResult
ControllerParams controller

# Trial information
int32 T  # Trajectory length
float64 frequency  # Controller frequency
int8[] state_datatypes  # Which data types to include in state
int8[] obs_datatypes # Which data types to include in observation
float64[] ee_points # A 3*n_points array containing offsets
float64[] ee_points_tgt # A 3*n_points array containing the desired ee_points for this trial
