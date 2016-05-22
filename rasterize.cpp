#include "rotator.h"
#include "uraster.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <cstdint>
#include <memory>
#include <array>
#include <functional>

typedef struct { float x,y,z,nx,ny,nz,s,t; } BunnyVert;

struct BunnyVertVsOut
{
	Eigen::Vector4f p;
	Eigen::Vector3f normal;
	Eigen::Vector2f texcoord;
	
	BunnyVertVsOut():
	p(0.0f,0.0f,0.0f,0.0f),normal(0.0f,0.0f,0.0f),texcoord(0.0f,0.0f)
	{}
	const Eigen::Vector4f& position() const
	{
		return p;
	}
	BunnyVertVsOut& operator+=(const BunnyVertVsOut& tp)
	{
		p+=tp.p;
		normal+=tp.normal;
		texcoord+=tp.texcoord;
		return *this;
	}
	BunnyVertVsOut& operator*=(const float& f)
	{
		p*=f;normal*=f;texcoord*=f;
		return *this;
	}
};

class BunnyPixel
{
public:
	Eigen::Vector3f diffuse_color;
	Eigen::Vector3f normal;
	float m_depth;
	float new_depth;
	float& depth() { return m_depth; }
	BunnyPixel():diffuse_color(0.0f,0.0f,0.0f),normal(0.0f,0.0f,0.0f),m_depth(-1e10f)
	{}
};

class TinyImage
{
public:
	std::shared_ptr<uint8_t> data;
	int width,height,n;
	TinyImage()
	{}
	
	TinyImage(const std::string& filename):data(stbi_load(filename.c_str(),&width,&height,&n,4),stbi_image_free)
	{}
	
	Eigen::Vector4f operator()(uint32_t si,uint32_t ti) const
	{
		if(data.get() == nullptr)
		{
			return Eigen::Vector4f(0.0f,0.0f,0.0f,0.0f);
		}
		const uint8_t* pix=data.get()+4*((ti % height)*width+(si % width));
		return Eigen::Vector4f(pix[0],pix[1],pix[2],pix[3])/255.0;
	}
	
	Eigen::Vector4f operator()(const float s,const float t) const
	{
		float sb=s*width;
		uint32_t sbi=floor(sb);
		float sf=sb-sbi;
		
		float tb=t*height;
		uint32_t tbi=floor(tb);
		float tf=tb-tbi;
		
		Eigen::Vector4f f00=TinyImage::operator()(sbi,tbi);
		Eigen::Vector4f f01=TinyImage::operator()(sbi,tbi+1);
		Eigen::Vector4f f10=TinyImage::operator()(sbi+1,tbi);
		Eigen::Vector4f f11=TinyImage::operator()(sbi+1,tbi+1);
		Eigen::Vector4f ftop=f00*(1.0f-sf)+f01*(sf);
		Eigen::Vector4f fbot=f10*(1.0f-sf)+f11*(sf);
		return ftop*(1.0f-tf)+fbot*(tf);
	}
};

BunnyVertVsOut example_vertex_shader(const BunnyVert& vin,const Eigen::Matrix4f& mvp)
{
	BunnyVertVsOut vout;
	vout.p=mvp*Eigen::Vector4f(vin.x,vin.y,vin.z,1.0f);
	//vout.p[3]=1.0f;
	vout.normal=Eigen::Vector3f(vin.nx,vin.ny,vin.nz);
	vout.texcoord=Eigen::Vector2f(vin.s,1.0-vin.t);
	
	//mn:-0.199768,
	//mx: 0.309477

	return vout;
}
BunnyPixel example_fragment_shader(const BunnyVertVsOut& fsin,const TinyImage& texture)
{
	BunnyPixel p;
	
	//p.color.head<3>()=fsin.color;
	p.diffuse_color=texture(fsin.texcoord.x(),fsin.texcoord.y()).head<3>();
	
	float f=fsin.p.z();
	Eigen::Vector2f zrange(-0.199768,0.309477);
	f-=zrange[0];
	f/=(zrange[1]-zrange[0]);
	//std::cerr << f << std::endl;
	//p.color[3]=f;
	//p.color=Eigen::Vector4f(f,f,f,f);
	
	p.normal=(fsin.normal/fsin.normal.norm())*0.5f+Eigen::Vector3f(0.5f,0.5f,0.5f);
	p.new_depth=f;
	//depth is already set by rasterizer
	return p;
}

template<class SelectorFunc>
void write_framebuffer(const uraster::Framebuffer<BunnyPixel>& fb,const SelectorFunc& sf,const std::string& filename)
{
	int num_channels=sf(fb(0,0)).size();
	uint8_t* pixels=new uint8_t[fb.width*fb.height*num_channels];
	std::unique_ptr<uint8_t[]> data(pixels);
	
	#pragma omp parallel for
	for(size_t i=0;i<fb.width*fb.height;i++)
	{
 		auto selected=sf(fb(i % fb.width,i / fb.width));
		for(int c=0;c<num_channels;c++)
		{
			pixels[num_channels*i+c]=std::max(0.0f,std::min(selected[c]*255.0f,255.0f));
		}
	}
	
	std::cerr << "Postprocessing complete.  Writing to file" << std::endl;
	if(0==stbi_write_png(filename.c_str(),fb.width,fb.height,num_channels,pixels,0))
	{
		std::cout << "Failure to write " << filename << std::endl;
	}
}

void rasterize(const std::vector<tinyobj::shape_t>& shapes,const std::vector<tinyobj::material_t>& materials,uint32_t width,uint32_t height,Eigen::Matrix4f camera_matrix,bool drawdepth)
{
	//camera_matrix=Eigen::Matrix4f::Identity();
	uraster::Framebuffer<BunnyPixel> tp(width,height);
	
	/*BunnyVert vertsin[3]={	{-1.0f,-1.0f,0.0f,	1.0f,0.0f,0.0f,	0.0f,0.0f},
				{0.0f,0.0f,0.0f,	0.0f,1.0f,0.0f,	0.0f,1.0f},
				{1.0f,-1.0f,0.0f,	0.0f,0.0f,1.0f,	1.0f,0.0f}};
	size_t indexin[3]={0,1,2};
	
	
	uraster::draw(tp,
		      vertsin,vertsin+3,
		      indexin,indexin+3,
		      (BunnyVertVsOut*)NULL,(BunnyVertVsOut*)NULL,
		      std::bind(example_vertex_shader,std::placeholders::_1,camera_matrix),
		      example_fragment_shader
	);*/
	Eigen::Vector3f mm=Eigen::Vector3f(100000000.0,100000000.0,100000000.0);
	Eigen::Vector3f mx=-Eigen::Vector3f(100000000.0,100000000.0,100000000.0);
	
	
	std::vector<TinyImage> images;
	for(const tinyobj::material_t& mat : materials)
	{
		std::cerr << "Loading " << mat.diffuse_texname << std::endl;
		images.emplace_back(mat.diffuse_texname);
	}
	
	for(const tinyobj::shape_t& shs : shapes)
	{
		std::vector<BunnyVert> vertsin(shs.mesh.positions.size()/3);
		for(size_t i=0;i<vertsin.size();i++)
		{
			BunnyVert& vert=vertsin[i];
			vert.x=shs.mesh.positions[3*i];
			vert.y=shs.mesh.positions[3*i+1];
			vert.z=shs.mesh.positions[3*i+2];
			
			Eigen::Vector4f tproj=camera_matrix*Eigen::Vector4f(vert.x,vert.y,vert.z,1.0);
			//std::cerr << tproj.z() << ",";
			
			mm.x()=std::min(tproj.x(),mm.x());mm.y()=std::min(tproj.y(),mm.y());mm.z()=std::min(tproj.z(),mm.z());
			mx.x()=std::max(tproj.x(),mx.x());mx.y()=std::max(tproj.y(),mx.y());mx.z()=std::max(tproj.z(),mx.z());
			
			vert.nx=-shs.mesh.normals[3*i];
			vert.ny=-shs.mesh.normals[3*i+1];
			vert.nz=-shs.mesh.normals[3*i+2];
			
			vert.s=shs.mesh.texcoords[2*i];
			vert.t=shs.mesh.texcoords[2*i+1];
		}
		std::vector<size_t> indices(shs.mesh.indices.cbegin(),shs.mesh.indices.cend());
		
		uraster::draw(tp,
			&vertsin[0],&vertsin[0]+vertsin.size(),
			&indices[0],&indices[0]+indices.size(),
			(BunnyVertVsOut*)NULL,(BunnyVertVsOut*)NULL,
			std::bind(example_vertex_shader,std::placeholders::_1,camera_matrix),
			std::bind(example_fragment_shader,std::placeholders::_1,images[shs.mesh.material_ids[0]])
		);
	}
	
	std::cerr << "mn:" << mm.z() << ",\n mx: " << mx.z() << std::endl;
	
 	//std::cerr << "mn':" << camera_matrix*Eigen::Vector4f(mm.x(),mm.y(),mm.z(),1.0) << ",\n mx': " << camera_matrix*Eigen::Vector4f(mx.x(),mx.y(),mx.z(),1.0) << std::endl;
	std::cerr << "Rendering complete.  Postprocessing." << std::endl;
	write_framebuffer(tp,[](const BunnyPixel& bp){ return bp.diffuse_color;},"out_diffuse.png");
	write_framebuffer(tp,[](const BunnyPixel& bp){ std::array<float,1> t; t[0]=bp.new_depth; return t;},"out_height.png");
	write_framebuffer(tp,[](const BunnyPixel& bp){ return bp.normal;},"out_normals.png");
	
}