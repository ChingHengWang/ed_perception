/*
* Author: Luis Fererira
* E-mail: luisfferreira@outlook.com
* Date: May 2015
*/

#include "entity_live_viewer_cv.h"
#include "../../plugins/shared_methods.h"

// ED data structures
#include <ed/error_context.h>
#include <ed/measurement.h>
#include <ed/io/filesystem/write.h>
#include <rgbd/Image.h>
#include <rgbd/serialization.h>
#include <tue/config/reader.h>

// C++ IO
#include <iostream>
#include <boost/filesystem.hpp>

// C++ sleep
#include <boost/thread/thread.hpp>

// ROS
#include <ros/init.h>

// ----------------------------------------------------------------------------------------------------


EntityLiveViewerCV::EntityLiveViewerCV(){
    window_margin_ = 20;
    preview_size_ = 400;
    max_age_ = 6;
    focused_idx_ = 0;
    state_freeze_ = false;
    module_name_ = "[Entity Live Viewer] ";
    model_name_ = "default";
    saved_measurements_ = 0;
    font_face_ = cv::FONT_HERSHEY_SIMPLEX;
    font_scale_ = 0.5;
    exit_ = false;
    sleep_interval_ = 80;

    client_.launchProbe("EntityViewerProbe", "libentity_viewer_probe.so");
    std::cout << module_name_ << "Probe '" << client_.probeName() <<"' launched" << std::endl;

    std::cout << module_name_ << "Ready!" << std::endl;
    std::cout << module_name_ << "How to use: " << std::endl;
    std::cout << module_name_ << "\t1 - 9 : Choose entity from the list " << std::endl;
    std::cout << module_name_ << "\tSPACE : Freeze the viewer" << std::endl;
    std::cout << module_name_ << "\tS : Store measurement" << std::endl;
    std::cout << module_name_ << "\tN : Change name used for the measurement" << std::endl;

    cv::namedWindow("Entity Live Viewer", cv::WINDOW_AUTOSIZE);
}

EntityLiveViewerCV::~EntityLiveViewerCV(){
    cv::destroyWindow("Entity Live Viewer");
}

// ----------------------------------------------------------------------------------------------------


void EntityLiveViewerCV::updateViewer(std::vector<viewer_common::EntityInfo>& entity_list){
    ed::ErrorContext errc("EntityLiveViewer -> updateViewer()");

    cv::Mat output_img = cv::Mat::zeros(900, 800, CV_8UC3);

    // set ROIs
    cv::Mat entity_preview_roi = output_img(cv::Rect(0, 0, preview_size_, preview_size_));
    cv::Mat entity_listroi = output_img(cv::Rect(0, entity_preview_roi.rows + window_margin_,
                                                  entity_preview_roi.cols + window_margin_, output_img.rows - entity_preview_roi.rows - window_margin_));
    cv::Mat entity_info_roi = output_img(cv::Rect(preview_size_ + window_margin_, 0,
                                                  output_img.cols - entity_preview_roi.cols - window_margin_, output_img.rows));

    // paint background white
    entity_preview_roi.setTo(cv::Scalar(255,255,255));
    entity_info_roi.setTo(cv::Scalar(30,30,30));
    entity_listroi.setTo(cv::Scalar(30,30,30));


    // ------------------ ENTITY IMAGE ----------------------

    // Draw entity masked image
    if (!entity_list.empty() && focused_idx_ < entity_list.size()){
        cv::Mat resized_img = cv::Mat::zeros(preview_size_, preview_size_, CV_8UC3);
        if (!entity_list[focused_idx_].masked_roi.empty()){
            resized_img = ed::perception::resizeSameRatio(entity_list[focused_idx_].masked_roi, preview_size_);
            resized_img.copyTo(entity_preview_roi);
        }
    }


    // ------------------ ENTITY INFO -----------------------

    cv::Point model_name_org(10, 30);
    cv::Point entity_list_org(10, 60);

    if (!entity_list.empty() && focused_idx_ < entity_list.size()){
        putTextMultipleLines(entity_list[focused_idx_].data_str, "\n", cv::Point(10,10), entity_info_roi);
    }

    // Draw model name
    std::stringstream model_name_info;
    model_name_info << "Model name: " << model_name_;
    cv::putText(entity_listroi, model_name_info.str(), model_name_org, font_face_, font_scale_, cv::Scalar(255,200, 200), 1, CV_AA);
    model_name_info.str("");
    model_name_info << "Saved counter: " << saved_measurements_;
    cv::putText(entity_listroi, model_name_info.str(), model_name_org + cv::Point(230, 0), font_face_, font_scale_, cv::Scalar(255,200, 200), 1, CV_AA);

    // -------------------ENTITY LIST ------------------------

    int vert_offset = 40;
    int counter = 1;


    // Draw entity list title
    std::stringstream list_title;
    list_title << "Entity List (" << entity_list.size() << "):";
    cv::putText(entity_listroi, list_title.str(), entity_list_org, font_face_, font_scale_+0.1, cv::Scalar(255,255, 255), 1, CV_AA);

    // Draw entity list
    for(std::vector<viewer_common::EntityInfo>::const_iterator entity_it = entity_list.begin(); entity_it != entity_list.end(); ++entity_it){

        std::stringstream ss;
        ss.precision(2);

        ss << "   " << counter << ": " << (std::string)(entity_it->id).substr(0, 4);

        // show type if it is known
//        if (!entity_it->type.empty())

        ss << "    [" << entity_it->type << "] [" << (double)entity_it->area << "]";

        // warning to show the entity was not updated
        if (entity_it->last_updated > 1)
            ss << " (!)";

        // show selected entity
        if (counter-1 == focused_idx_)
            ss << "  <--";

        // draw text
        cv::putText(entity_listroi, ss.str(), entity_list_org + cv::Point(0, vert_offset), font_face_, font_scale_, cv::Scalar(255,255,255), 1, CV_AA);
        vert_offset += 20;
        counter++;
    }

    // ------------------------------------------------------

    if (!state_freeze_){
        // update age of the entities in the list, and remove old ones
        std::vector<viewer_common::EntityInfo>::iterator it = entity_list.begin();
        while(it != entity_list.end()) {
            if(it->last_updated > max_age_){
                it = entity_list.erase(it);
            }else{
                it->last_updated++;
                ++it;
            }
        }
    }

    // show viewer
    cv::imshow("Entity Live Viewer", output_img);
}


// ----------------------------------------------------------------------------------------------------


void EntityLiveViewerCV::processKeyPressed(char key, std::vector<viewer_common::EntityInfo>& entity_list){
    ed::ErrorContext errc("EntityLiveViewer -> processKeyPressed()");

    // return if no key was pressed
    if(key == -1) return;

    std::cout << module_name_ << "Key pressed = " << key << " (" << key % 255 << ')' << std::endl;

    switch (key){
        // choose entity from the list
        case '1':   if (entity_list.size() >= 1) focused_idx_ = 0;
        break;
        case '2':   if (entity_list.size() >= 2) focused_idx_ = 1;
        break;
        case '3':   if (entity_list.size() >= 3) focused_idx_ = 2;
        break;
        case '4':   if (entity_list.size() >= 4) focused_idx_ = 3;
        break;
        case '5':   if (entity_list.size() >= 5) focused_idx_ = 4;
        break;
        case '6':   if (entity_list.size() >= 6) focused_idx_ = 5;
        break;
        case '7':   if (entity_list.size() >= 7) focused_idx_ = 6;
        break;
        case '8':   if (entity_list.size() >= 8) focused_idx_ = 7;
        break;
        case '9':   if (entity_list.size() >= 9) focused_idx_ = 8;
        break;

        // next and previous entity
        case 84:   if (focused_idx_ < entity_list.size()-1) focused_idx_++;
        break;
        case 82:   if (focused_idx_ > 0) focused_idx_--;
        break;

        // toggle freeze viewer
        case ' ':   state_freeze_ = !state_freeze_;
                    std::cout << module_name_ << "Viewer paused: " << state_freeze_ << std::endl;
        break;

        // store measurement
        case 's':   if (focused_idx_ <= entity_list.size() || entity_list[focused_idx_].last_updated == 0){
                        requestStoreMeasurement(entity_list[focused_idx_].id, model_name_);
                    }else
                        std::cout << module_name_ << "Entity selected is no longer available. Select another." << key << std::endl;
        break;

        // change model name
        case 'n':   std::cout << module_name_ << "Model name: ";
                    std::cin >> model_name_;
                    saved_measurements_ = 0;
        break;

        // Close viewer
        case 'q':   std::cout << module_name_ << "Quitting viewer!" << std::endl;
                    exit_ = true;
        break;

        // Close viewer (ESC)
        case 27:   std::cout << module_name_ << "Quitting viewer!" << std::endl;
                   exit_ = true;
        break;
    }
}


// ----------------------------------------------------------------------------------------------------


void EntityLiveViewerCV::putTextMultipleLines(const std::string& text, const std::string& delimiter, cv::Point origin, cv::Mat& image_out){
    ed::ErrorContext errc("EntityLiveViewer -> putTextMultipleLines()");

    int pos = 0;
    std::string token;
    cv::Point line_spacing(0, 15);
    int counter = 1;

    std::string text_bckp = text;

    // split the string by the delimiter
    while ((pos = text_bckp.find(delimiter)) != std::string::npos) {
        token = text_bckp.substr(0, pos);
        text_bckp.erase(0, pos + delimiter.length());

        cv::putText(image_out, token, origin + line_spacing*counter, font_face_, font_scale_-0.1, cv::Scalar(255,255, 255), 1, CV_AA);
//        std::cout << token << std::endl;
        counter++;
    }
}


// ----------------------------------------------------------------------------------------------------


int EntityLiveViewerCV::requestStoreMeasurement(const std::string& entity_id, const std::string& model_name){
    ed::ErrorContext errc("EntityLiveViewer -> storeMeasurement()");

    tue::serialization::Archive req;
    tue::serialization::Archive res;

    req << (int)viewer_common::STORE_MEASUREMENT;
    req << entity_id;
    req << model_name;

    // send request to client
    if (client_.process(req, res)){

        int result;
        res >> result;

        if (result == 0){
            std::cout << module_name_ << "Measurement saved with model name '" << model_name << "'" << std::endl;
            saved_measurements_++;
        }else
            std::cout << module_name_ << "Problem saving measurement!" << std::endl;

        return 0;
    } else {
        std::cout << module_name_ << "Probe request failed!" << std::endl;
        return 1;
    }
}


// ----------------------------------------------------------------------------------------------------


int EntityLiveViewerCV::requestEntityROI(const std::string& entity_id, cv::Mat& roi){

    tue::serialization::Archive req;
    tue::serialization::Archive res;

    req << (int)viewer_common::GET_ENTITY_ROI;
    req << entity_id;

    // send request to client
    if (client_.process(req, res)){

        int cols,rows;
        res >> rows;
        res >> cols;

        // std::cout << module_name_ << "size : (" << rows << "x" << cols << ")" << std::endl;

        // TEMPORARY ugly bug fix, sometimes the size is wrong for some reason
        if (rows * cols > 0 && rows * cols < 1280*1024 && rows > 1 && cols > 1){

            roi = cv::Mat::zeros(rows, cols, CV_8UC3);

            int size = cols * rows * 3;
            for(int i = 0; i < size; ++i)
                res >> roi.data[i];

            if (roi.empty()){
                std::cout << module_name_ << "Received empty Mat!" << std::endl;
            }
        }else if (rows * cols == 0){
            std::cout << module_name_ << "Probe could not find entity ID!" << std::endl;
        }else
            std::cout << module_name_ << "Entity image size incorrect! (" << rows << "x" << cols << ")" << std::endl;

        return 0;
    } else {
        std::cout << module_name_ << "Probe request failed!" << std::endl;
        return 1;
    }

}


// ----------------------------------------------------------------------------------------------------


int EntityLiveViewerCV::requestEntityData(const std::string& entity_id, std::string& data){

    tue::serialization::Archive req;
    tue::serialization::Archive res;

    req << (int)viewer_common::GET_ENTITY_DATA;
    req << entity_id;

    // send request to client
    if (client_.process(req, res)){
        res >> data;
//        std::cout << module_name_ << "got data with length " << data.size() << std::endl;
        return 0;
    } else {
        std::cout << module_name_ << "Probe request failed!" << std::endl;
        return 1;
    }
}

// ----------------------------------------------------------------------------------------------------


int EntityLiveViewerCV::requestEntityList(std::vector<viewer_common::EntityInfo>& list){

    tue::serialization::Archive req;
    tue::serialization::Archive res;

    req << (int)viewer_common::GET_ENTITY_LIST;

    // send request to client
    if (client_.process(req, res)){
        int num_entities;

        res >> num_entities;

        for(int i=0 ; i<num_entities; i++){
            std::string id = "";
            std::string type = "";
            std::string data = "";
            double area;

            res >> id;
            res >> type;
            res >> data;
            res >> area;

            list.push_back(viewer_common::EntityInfo(id, type, data, area));
            // std::cout << "pushing " << id << " with area " << area << std::endl;
        }

        return 0;
    } else {
        std::cout << module_name_ << "Probe request failed!" << std::endl;
        return 1;
    }
}


// ----------------------------------------------------------------------------------------------------


int EntityLiveViewerCV::mainLoop(){

    char key_pressed = -1;
    std::vector<viewer_common::EntityInfo> entity_list;
    std::vector<viewer_common::EntityInfo> updated_list;

    while(!exit_){

        if (!state_freeze_){
            updated_list.clear();
            requestEntityList(updated_list);

            // update the current entity info or create a new one
            for(std::vector<viewer_common::EntityInfo>::iterator new_entity_it = updated_list.begin(); new_entity_it != updated_list.end(); ++new_entity_it){
                bool exists = false;
                for(std::vector<viewer_common::EntityInfo>::iterator entity_it = entity_list.begin(); entity_it != entity_list.end(); ++entity_it){
                    if (entity_it->id.compare(new_entity_it->id) == 0){
                        entity_it->last_updated = 0;
                        entity_it->data_str = new_entity_it->data_str;
                        entity_it->area = new_entity_it->area;
                        exists = true;
                    }
                }

                if (!exists){
                    entity_list.push_back(viewer_common::EntityInfo(new_entity_it->id, new_entity_it->type, new_entity_it->data_str, new_entity_it->area));
                }
            }

            if (!entity_list.empty() && entity_list[focused_idx_].last_updated == 0){
//                std::cout << "focus: " << focused_idx_ << ", id: " << entity_list[focused_idx_].id << std::endl;
                // update the entity image
                requestEntityData(entity_list[focused_idx_].id, entity_list[focused_idx_].data_str);
                requestEntityROI(entity_list[focused_idx_].id, entity_list[focused_idx_].masked_roi);
            }
        }

        // process key presses
        key_pressed = cv::waitKey(10);
        processKeyPressed(key_pressed, entity_list);

        // re-draw the viewer
        updateViewer(entity_list);

        // sleep
        boost::this_thread::sleep(boost::posix_time::milliseconds(sleep_interval_));

    }
    return 0;
}


// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------


int main( int argc, char* argv[] )
{
    std::vector<viewer_common::EntityInfo> entity_list;

    ros::init(argc, argv, "entity_viewer_cv");

    EntityLiveViewerCV viewer;

    return viewer.mainLoop();
}
