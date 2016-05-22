#ifndef ROTATOR_H
#define ROTATOR_H

#include "tiny_obj_loader.h"
#include<Eigen/Dense>
#include<vector>

struct RotatorStats
{
	Eigen::Vector3d minval;
	Eigen::Vector3d maxval;
	Eigen::Matrix4d cameramat;
};

RotatorStats get_pca(std::vector<tinyobj::shape_t>& shapes,const double scalexy=1.0,const double scalez=1.0,const double rotxy=0.0);
#endif