#include "rotator.h"
#include<Eigen/SVD>
#include<algorithm>
using namespace Eigen;

RotatorStats get_pca(std::vector<tinyobj::shape_t>& shapes,const double scalexy,const double scalez,const double rotxy)
{
	size_t num_points=0;
	for(const tinyobj::shape_t& shs : shapes)
	{
		num_points+=shs.mesh.positions.size()/3;
	}
	
	Eigen::MatrixXd points(3,num_points);
	
	num_points=0;
	double* databuf=points.data();
	for(const tinyobj::shape_t& shs : shapes)
	{
		std::copy(shs.mesh.positions.cbegin(),shs.mesh.positions.cend(),databuf);
		databuf+=shs.mesh.positions.size();
	}
	Vector3d mn=points.rowwise().mean();
	
	for(size_t ci=0;ci<num_points;ci++)
	{
		points.col(ci)-=mn;
	}
	JacobiSVD<MatrixXd> svd(points, ComputeThinU | ComputeThinV);
	//A=U,S,V'
	Matrix4d m;
	m.col(3)=Vector4d(0.0,0.0,0.0,1.0);
	m.row(3)=RowVector4d(0.0,0.0,0.0,1.0);
	m.block<3,3>(0,0)=svd.matrixU().transpose();
	m.block<3,1>(0,3)=-m.block<3,3>(0,0)*mn;
	
	
	points=svd.singularValues().asDiagonal()*(svd.matrixV().transpose());
	
	RotatorStats rstats;
	rstats.maxval=points.rowwise().maxCoeff();
	rstats.minval=points.rowwise().minCoeff();	
	rstats.cameramat=m;
	double scxy=0.0;
	for(int i=0;i<3;i++)
	{
		scxy=std::max(scxy,rstats.maxval[i]);
		scxy=std::max(scxy,-rstats.minval[i]);
	}
	
	Vector4d scale(1.0/scxy,1.0/scxy,1.0/scxy,1.0);
	rstats.cameramat=Eigen::Vector4d(scalexy/scxy,scalexy/scxy,scalez/scxy,1.0).asDiagonal()*rstats.cameramat;
	float s=sin(rotxy),c=cos(rotxy);
	Matrix4d rot;
	rot <<  c, s,0.0,0.0,
		-s,c,0.0,0.0,
		0.0,0.0,1.0,0.0,
		0.0,0.0,0.0,1.0;
	rstats.cameramat=rot*rstats.cameramat;
	
	/*
	
	num_points=0;
	databuf=points.data();
	for(tinyobj::shape_t& shs : shapes)
	{
		std::copy(databuf,databuf+shs.mesh.positions.size(),shs.mesh.positions.begin());
		databuf+=shs.mesh.positions.size();
	}*/
	
	return rstats;
}
