#include "gps_agent_pkg/sample.h"
#include "gps/proto/gps.pb.h"
#include "ros/ros.h"

using namespace gps_control;

Sample::Sample(int T)
{
	ROS_INFO("Initializing Sample with T=%d", T);
	T_ = T;
	internal_data_size_.resize((int)gps::TOTAL_DATA_TYPES);
	internal_data_format_.resize((int)gps::TOTAL_DATA_TYPES);
	meta_data_.resize((int)gps::TOTAL_DATA_TYPES);
	// Fill in all possible sample types
	for(int i=0; i<gps::TOTAL_DATA_TYPES; i++){
		internal_data_[(gps::SampleType)i].resize(T);
		internal_data_size_[i] = -1; //initialize to -1
	}
  ROS_INFO("done sample constructor");
}

Sample::~Sample()
{
}

void* Sample::get_data_pointer(int t, gps::SampleType type)
{
    return NULL;
}

void Sample::set_data_vector(int t, gps::SampleType type, double *data, int data_size, SampleDataFormat data_format)
{
    set_data_vector(t,type,data,data_size,1,data_format);
}

void Sample::set_data_vector(int t, gps::SampleType type, double *data, int data_rows, int data_cols, SampleDataFormat data_format)
{
    if(t >= T_) ROS_ERROR("Out of bounds t: %d/%d", t, T_);
    if (data_format == SampleDataFormatEigenVector) {
        Eigen::VectorXd &vector = boost::get<Eigen::VectorXd>(internal_data_[type][t]);
        if (vector.rows() != data_rows || data_cols != 1)
            ROS_ERROR("Invalid size in set_data_vector! %i vs %i and cols %i for type %i",
                vector.rows(), data_rows, data_cols, (int)type);
        memcpy(vector.data(), data, sizeof(double) * data_rows * data_cols);
    }
    else if (data_format == SampleDataFormatEigenMatrix) {
        Eigen::MatrixXd &matrix = boost::get<Eigen::MatrixXd>(internal_data_[type][t]);
        if (matrix.rows() != data_rows || matrix.cols() != data_cols)
            ROS_ERROR("Invalid size in set_data_vector! %i vs %i and %i vs %i for type %i",
                matrix.rows(), data_rows, matrix.cols(), data_cols, (int)type);
        memcpy(matrix.data(), data, sizeof(double) * data_rows * data_cols);
    }
    else {
        ROS_ERROR("Cannot use set_data_vector with non-Eigen types! Use set_data instead.");
    }
    return;
}

void Sample::set_data(int t, gps::SampleType type, SampleVariant data, int data_size, SampleDataFormat data_format)
{
    if(t >= T_) ROS_ERROR("Out of bounds t: %d/%d", t, T_);
    internal_data_[type][t] = data;
    return;
}

void Sample::get_data(int t, gps::SampleType type, void *data, int data_size, SampleDataFormat data_format) const
{
    ROS_ERROR("Not supported!");
    return;
}

void Sample::set_meta_data(gps::SampleType type, int data_size, SampleDataFormat data_format, OptionsMap meta_data)
{
    // A simplified version of set_meta_data for non-matrix types.
    set_meta_data(type, data_size, 1, data_format, meta_data);
}

void Sample::set_meta_data(gps::SampleType type, int data_size_rows, int data_size_cols, SampleDataFormat data_format, OptionsMap meta_data)
{
    int type_key = (int) type;
    internal_data_size_[type_key] = data_size_rows * data_size_cols;
    internal_data_format_[type_key] = data_format;
    meta_data_[type_key] = meta_data;
    // If this is a matrix or vector type, preallocate it now for fast copy later.
    if (data_format == SampleDataFormatEigenVector)
    {
        for (int t = 0; t < T_; t++)
            internal_data_[type][t] = Eigen::VectorXd(data_size_rows);
    }
    if (data_format == SampleDataFormatEigenMatrix)
    {
        for (int t = 0; t < T_; t++)
            internal_data_[type][t] = Eigen::MatrixXd(data_size_rows, data_size_cols);
    }
    return;
}

void Sample::get_available_dtypes(std::vector<gps::SampleType> &types){
    for(int i=0; i<gps::TOTAL_DATA_TYPES; i++){
        if(internal_data_size_[i] != -1){
            types.push_back((gps::SampleType)i);
        }
    }
}

void Sample::get_meta_data(gps::SampleType type, int &data_size, SampleDataFormat &data_format, OptionsMap &meta_data_) const
{
    ROS_ERROR("Not implemented!");
    return;
}

void Sample::get_state(int t, Eigen::VectorXd &x) const
{
	x.fill(0.0);
    return;
}

void Sample::get_obs(int t, Eigen::VectorXd &obs) const
{
	obs.fill(0.0);
    return;
}

void Sample::get_data_all_timesteps(Eigen::VectorXd &data, gps::SampleType datatype){
	int size = internal_data_size_[(int)datatype];
	data.resize(size*T_);
	std::vector<gps::SampleType> dtype_vector;
	dtype_vector.push_back(datatype);

	Eigen::VectorXd tmp_data;
	for(int t=0; t<T_; t++){
		get_data(t, tmp_data, dtype_vector);
		// Fill in original data
		for(int i=0; i<size; i++){
			data[t*size+i] = tmp_data[i];
		}
	}
}

void Sample::get_data(int T, Eigen::VectorXd &data, gps::SampleType datatype){
	int size = internal_data_size_[(int)datatype];
	data.resize(size*T);
	std::vector<gps::SampleType> dtype_vector;
	dtype_vector.push_back(datatype);

	Eigen::VectorXd tmp_data;
	for(int t=0; t<T; t++){
		get_data(t, tmp_data, dtype_vector);
		// Fill in original data
		for(int i=0; i<size; i++){
			data[t*size+i] = tmp_data[i];
		}
	}
}

void Sample::get_shape(gps::SampleType sample_type, std::vector<int> &shape)
{
    int dtype = (int)sample_type;
    int size = internal_data_size_[dtype];
    shape.clear();
    if(internal_data_format_[dtype] == SampleDataFormatEigenVector){
        shape.push_back(size);
    }else if (internal_data_format_[dtype] == SampleDataFormatEigenMatrix){
        // Grab shape from first entry at T=0
        Eigen::MatrixXd &sensor_data = boost::get<Eigen::MatrixXd>(internal_data_[sample_type][0]);
        shape.push_back(sensor_data.rows());
        shape.push_back(sensor_data.cols());
    }
}

void Sample::get_data(int t, Eigen::VectorXd &data, std::vector<gps::SampleType> datatypes)
{
	if(t >= T_) ROS_ERROR("Out of bounds t: %d/%d", t, T_);
    // Calculate size
    int total_size = 0;
	for(int i=0; i<datatypes.size(); i++){
		int dtype = (int)datatypes[i];
		if(dtype >= internal_data_size_.size()){
			ROS_ERROR("Requested size of dtype %d, but internal_data_size_ only has %d elements", dtype,
				internal_data_size_.size());
		}
		total_size += internal_data_size_[dtype];
	}

	data.resize(total_size);
	data.fill(0.0);

    // Fill in data
    int current_idx = 0;
	for(int i=0; i<datatypes.size(); i++){
		int dtype = (int)datatypes[i];
		if(dtype >= internal_data_.size()){
			ROS_ERROR("Requested internal data of dtype %d, but internal_data_ only has %d elements", dtype,
				internal_data_.size());
		}
		const SampleList &sample_list = internal_data_[datatypes[i]];
		const SampleVariant &sample_variant = sample_list[t];
		int size = internal_data_size_[dtype];

		//Handling for specific datatypes
		if(internal_data_format_[dtype] == SampleDataFormatEigenVector){
			const Eigen::VectorXd &sensor_data = boost::get<Eigen::VectorXd>(sample_variant);
			data.segment(current_idx, size) = sensor_data;
			current_idx += size;
		}else if (internal_data_format_[dtype] == SampleDataFormatEigenMatrix){
			Eigen::MatrixXd sensor_data = boost::get<Eigen::MatrixXd>(sample_variant).transpose();
            Eigen::VectorXd flattened_mat(Eigen::Map<Eigen::VectorXd>(sensor_data.data(), sensor_data.size()));
			flattened_mat.resize(sensor_data.cols()*sensor_data.rows(), 1);
			data.segment(current_idx, size) = flattened_mat;
			current_idx += size;
		}else {
			ROS_ERROR("Datatypes currently must be in Eigen::Vector/Eigen::Matrix format. Offender: dtype=%d", dtype);
		}

	}

	return;
}

int Sample::get_T(){
	return T_;
}


void Sample::get_action(int, Eigen::VectorXd &u) const
{
    return;
}

