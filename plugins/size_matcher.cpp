/*
* Author: Luis Fererira
* E-mail: luisfferreira@outlook.com
* Date: January 2015
*/

#include "size_matcher.h"

#include "ed/measurement.h"
#include "ed/entity.h"
#include <ed/error_context.h>

#include <rgbd/Image.h>
#include <rgbd/View.h>

// Loading models
#include <fstream>


// NOTE: currently only looks at hight!

// ----------------------------------------------------------------------------------------------------

SizeMatcher::SizeMatcher() :
    Module("size_matcher"),
    init_success_(false)
{
}

// ----------------------------------------------------------------------------------------------------

SizeMatcher::~SizeMatcher()
{
}

// ----------------------------------------------------------------------------------------------------


void SizeMatcher::configure(tue::Configuration config) {

    if (!config.value("size_diff_threshold", size_diff_threshold_, tue::OPTIONAL))
        std::cout << "[" << module_name_ << "] " << "Parameter 'size_diff_threshold' not found. Using default: " << size_diff_threshold_ << std::endl;

    if (!config.value("small_size_treshold", size_diff_threshold_, tue::OPTIONAL))
        std::cout << "[" << module_name_ << "] " << "Parameter 'small_size_treshold' not found. Using default: " << small_size_treshold_ << std::endl;

    if (!config.value("medium_size_treshold", medium_size_treshold_, tue::OPTIONAL))
        std::cout << "[" << module_name_ << "] " << "Parameter 'medium_size_treshold' not found. Using default: " << medium_size_treshold_ << std::endl;
}

// ----------------------------------------------------------------------------------------------------

void SizeMatcher::loadConfig(const std::string& config_path) {

    module_name_ = "size_matcher";

    // default values in case configure(...) is not called!
    size_diff_threshold_ = 0.8;
    small_size_treshold_ = 0.5;
    medium_size_treshold_ = 0.7;

    init_success_ = true;

    std::cout << "[" << module_name_ << "] " << "Ready!"<< std::endl;
}

// ----------------------------------------------------------------------------------------------------

void SizeMatcher::loadModel(const std::string& model_name, const std::string& model_path)
{

    std::string models_folder = model_path.substr(0, model_path.find_last_of("/") - 1); // remove last slash
    models_folder = models_folder.substr(0, models_folder.find_last_of("/"));   // remove size_matcher from path

    std::string path = models_folder + "/models/" + model_name +  "/" +  model_name + ".yml";

    if (loadLearnedModel(path, model_name)){
//        std::cout << "[" << module_name_ << "] " << "Loaded sizes for " << model_name << std::endl;
    }
    else{
//        std::cout << "[" << module_name_ << "] " << "Couldn not load sizes for " << path << "!" << std::endl;
    }
}


// ----------------------------------------------------------------------------------------------------

void SizeMatcher::process(const ed::perception::WorkerInput& input, ed::perception::WorkerOutput& output) const
{
    const ed::EntityConstPtr& e = input.entity;
    tue::Configuration& result = output.data;

    ed::ErrorContext errc("Processing entity in SizeMatcher");

    if (!init_success_)
        return;


    const ed::ConvexHull2D& chull = e->convexHull();

    double object_height = chull.height();
    double object_width = 0;

    std::map<std::string, double> hypothesis;

    // compare object size to loaded models
    for(std::map<std::string, std::vector<ObjectSize> >::const_iterator it = models_.begin(); it != models_.end(); ++it)
    {
        const std::string& label = it->first;
        const std::vector<ObjectSize>& sizes = it->second;

        double best_score = 0;

        for(std::vector<ObjectSize>::const_iterator it_size = sizes.begin(); it_size != sizes.end(); ++it_size)
        {
            const ObjectSize& model_size = *it_size;

            // Temp
            double diff_w = 0; //std::abs(model_size.width - object_width);
            double diff_h = std::abs(model_size.height - object_height);

            double max_diff = 0.1; // TODO: magic number

            double score;
            if (diff_w > max_diff || diff_h > max_diff)
                score = 0;
            else
            {
                double w_score = 1.0 - (diff_w / max_diff); // TODO: magic score function
                double h_score = 1.0 - (diff_h / max_diff); // TODO: magic score function
                score = (w_score + h_score) / 2;
            }

            if (score > best_score)
                object_width = model_size.width;

            best_score = std::max(best_score, score);
        }

//        std::cout << label << ": " << best_score << std::endl;

        // if the size difference between models is less than threshold, use the model
        hypothesis[label] = 0.5 * best_score; // TODO: magic number
    }

//    std::cout << "Size matcher: " << e->id() << ": " << object_width << ", " << object_height << std::endl;

    if (object_height < 1)
        hypothesis["human"] = 0;

    // ----------------------- SAVE RESULTS -----------------------

    // create group if it doesnt exist
    if (!result.readGroup("perception_result", tue::OPTIONAL))
    {
        result.writeGroup("perception_result");
    }

    result.writeGroup("size_matcher");

    result.writeGroup("size");
    result.setValue("width", object_width);
    result.setValue("height", object_height);
    result.endGroup();

    if ((object_width + object_height) < small_size_treshold_){
        result.setValue("label", "small_size");
    }else if (small_size_treshold_ < (object_width + object_height) && (object_width + object_height) < medium_size_treshold_){
        result.setValue("label", "medium_size");
    }else if (medium_size_treshold_ < (object_width + object_height)){
        result.setValue("label", "large_size");
    }

    result.setValue("score", 1.0);

    output.type_update.setUnknownScore(0.1); // TODO: magic number

    // assert hypothesis
    if (!hypothesis.empty()){
        result.writeArray("hypothesis");
        for (std::map<std::string, double>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it)
        {
            result.addArrayItem();
            result.setValue("name", it->first);
            result.setValue("score", std::max(it->second, 0.0));
            result.endArrayItem();

            output.type_update.setScore(it->first, std::max(it->second, 0.0));
        }
        result.endArray();
    }

    result.endGroup();  // close size_matcher group
    result.endGroup();  // close perception_result group
}

// ----------------------------------------------------------------------------------------------------

bool SizeMatcher::loadLearnedModel(std::string path, std::string model_name){
    if (path.empty())
    {
        std::cout << "[" << module_name_ << "] " << "Empty path!" << path << std::endl;
        return false;
    }

    tue::Configuration conf;
    float width;
    float height;
    std::vector<ObjectSize> model_sizes;

    if (!conf.loadFromYAMLFile(path))
    {
//        std::cout << "Could not load " << path << std::endl;
        return false;
    }

    if (!conf.readGroup("model"))
    {
        std::cout << "[" << module_name_ << "] " << "Could not find 'model' group" << std::endl;
        return false;
    }

    if (conf.readArray("size")) // read Size arary
    {
        while(conf.nextArrayItem())
        {
            if (conf.value("height", height, tue::OPTIONAL) && conf.value("width", width, tue::OPTIONAL))  // read height and width
            {
                ObjectSize obj_sz(width, height);
                model_sizes.push_back(obj_sz);
            }
            else
                std::cout << "[" << module_name_ << "] " << "Could not find 'height' and 'width' values" << std::endl;
        }

        if (!model_sizes.empty())  // save sizes to map
            models_[model_name] = model_sizes;
        else
            std::cout << "[" << module_name_ << "] " << "Could not read any sizes" << std::endl;

        conf.endArray();    // close Size array
    }else
        std::cout << "[" << module_name_ << "] " << "Could not find 'size' group" << std::endl;

    conf.endGroup();  // close Model group

    return true;
}

ED_REGISTER_PERCEPTION_MODULE(SizeMatcher)

