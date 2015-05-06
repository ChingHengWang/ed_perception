
#ifndef ENTITY_LIVE_VIEWER_H_
#define ENTITY_LIVE_VIEWER_H_

// Show measurement
#include <opencv2/highgui/highgui.hpp>

// ED
#include <ed/entity.h>
#include <ed/world_model.h>
#include <ed/update_request.h>
#include <ed/perception/module.h>

// ----------------------------------------------------------------------------------------------------

class EntityInfo{

public:
    ed::EntityConstPtr e;
    std::string id;
    cv::Mat color_img;
    cv::Mat depth_img;
    cv::Mat masked_roi;
    cv::Mat mask;
    cv::Rect roi;
    tue::config::DataConstPointer data;

    EntityInfo(const ed::EntityConstPtr& e)
        : id(e->id().str()), data(e->data()){}
};

class EntityLiveViewer{
    public:

       EntityLiveViewer();

       // FUNCTIONS
       void addEntity(const ed::EntityConstPtr& entity);
       void updateViewer();

    private:

        // CONFIGURATION VARIABLES
        std::string module_name_;
        std::string debug_dir_;
        std::vector<EntityInfo> entity_list_;

        int key_pressed_;

        int window_margin_;
        int preview_size_;

        void processKeyPressed(int key);
        void myDrawMultiLineText(std::string InputParagraph, cv::Point Origin);
};

#endif
