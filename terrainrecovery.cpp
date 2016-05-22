#include<iostream>
#include<vector>
#include "tiny_obj_loader.h"
#include "rotator.h"
using namespace std;

void rasterize(const std::vector<tinyobj::shape_t>& shapes,const std::vector<tinyobj::material_t>& materials,uint32_t width,uint32_t height,Eigen::Matrix4f camera_matrix,bool drawdepth=false);

int main(int argc,char** argv)
{
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	
	if(!tinyobj::LoadObj(shapes,materials,err,argv[1],NULL,tinyobj::triangulation | tinyobj::calculate_normals))
	{
		cerr << "Error loading model " << argv[1] << ":\n\t" << err << endl;
		return -1;
	}
	
	RotatorStats rstats=get_pca(shapes,1.0,1.0,-M_PI/12.0);
	cout << rstats.cameramat << endl;
	cout << rstats.minval << endl;
	cout << rstats.maxval << endl;
	rasterize(shapes,materials,1024*16,1024*16,rstats.cameramat.cast<float>(),false);
	
	return 0;
}