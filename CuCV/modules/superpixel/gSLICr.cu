#include "gSLICr.h"
#include <math.h>

using namespace std;
using namespace CuCv;

/** ------------------------------------------------------ **/
gSLICr::gSLICr(const Settings &settings)
    : settings_(settings),
      
      source_image_(settings._image_size.x,
                    settings._image_size.y),
      
      index_image_(settings._image_size.x, 
                   settings._image_size.y,
                   true,
                   true,
                   true,
                   cudaFilterModePoint),
      
      accum_map_(settings._image_size.x * settings._image_size.y)
{
    if (settings_._size_control_method == GIVEN_NUM)
	{
		float cluster_size = (float)(settings_._image_size.x * settings_._image_size.y) / (float) settings_._num_segmentations;
		spixel_size_ = (int)ceil(sqrt(cluster_size)); 
	}
	else
	{
        spixel_size_ = settings_._spixel_size;
	}
}

/** ------------------------------------------------------ **/
void
gSLICr::convertColor()
{
    
}


/** ------------------------------------------------------ **/
void
gSLICr::initClusterCenters()
{
    
}


/** ------------------------------------------------------ **/
void
gSLICr::findCenterAssociations()
{
    
}


/** ------------------------------------------------------ **/
void
gSLICr::updateClusterCenters()
{
    
}


/** ------------------------------------------------------ **/
void
gSLICr::enforceConnectivity()
{
    
}

/** ------------------------------------------------------ **/
gSLICr::IndexImage &
gSLICr::segment(void * image_data)
{
    source_image_.loadData(image_data, CPU_TO_GPU);
    convertColor();
    
    initClusterCenters();
    findCenterAssociations();
    
    for (int i=0;i<settings_._num_iterations;i++)
    {
        updateClusterCenters();
        findCenterAssociations();
    }
    
    if (settings_._enforce_connectivity)
        enforceConnectivity();
    
    return index_image_;
}


/** ------------------------------------------------------ **/
void
gSLICr::drawSegmentationOverlay(void * image_data)
{
    
}
