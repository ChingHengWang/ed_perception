/*
* Author: Luis Fererira
* E-mail: luisfferreira@outlook.com
* Date: January 2015
*/

#include "open_biometrics_ed.h"

#include "ed/measurement.h"
#include <ed/entity.h>

#include <rgbd/Image.h>
#include <rgbd/View.h>

#include <boost/filesystem.hpp>

// ----------------------------------------------------------------------------------------------------

OpenBrEd::OpenBrEd() :
    PerceptionModule("open_biometrics_ed"),
    init_success_(false)
{
}


// ----------------------------------------------------------------------------------------------------


OpenBrEd::~OpenBrEd()
{
    br::Context::finalize();
}


// ----------------------------------------------------------------------------------------------------


void OpenBrEd::configure(tue::Configuration config) {

    if (!config.value("debug_mode", debug_mode_, tue::OPTIONAL))
        std::cout << "[" << module_name_ << "] " << "Parameter 'debug_mode' not found. Using default: " << debug_mode_ << std::endl;

    if (!config.value("debug_folder", debug_folder_, tue::OPTIONAL))
        std::cout << "[" << module_name_ << "] " << "Parameter 'debug_folder' not found. Using default: " << debug_folder_ << std::endl;

    if (debug_mode_){
        // clean the debug folder if debugging is active
        try {
            boost::filesystem::path dir(debug_folder_);
            boost::filesystem::remove_all(dir);
            boost::filesystem::create_directories(dir);
        } catch(const boost::filesystem::filesystem_error& e){
           if(e.code() == boost::system::errc::permission_denied)
               std::cout << "[" << module_name_ << "] " << "boost::filesystem permission denied" << std::endl;
           else
               std::cout << "[" << module_name_ << "] " << "boost::filesystem failed with error: " << e.code().message() << std::endl;
        }

        // create debug window
        cv::namedWindow("Open Biometrics ED Output", CV_WINDOW_AUTOSIZE);
    }

    init_success_ = true;

    std::cout << "[" << module_name_ << "] " << "Ready!" << std::endl;
}


// ----------------------------------------------------------------------------------------------------


void OpenBrEd::loadConfig(const std::string& config_path) {

    module_name_ = "open_br_ed";
    module_path_ = config_path;
    debug_folder_ = "/tmp/open_br_ed/";
    debug_mode_ = false;

    int argc = 1;
    char* n_argv[] = {"param0"};

    br::Context::initialize(argc, n_argv);

    // Retrieve class for enrolling templates later
    br_age_estimation = br::Transform::fromAlgorithm("AgeEstimation");
    br_gender_estimation = br::Transform::fromAlgorithm("GenderEstimation");
    br_face_rec = br::Transform::fromAlgorithm("FaceRecognition");
//    br_face_rec_dist = br::Transform::fromAlgorithm("FaceRecognition");
}


// ----------------------------------------------------------------------------------------------------


void OpenBrEd::process(ed::EntityConstPtr e, tue::Configuration& config) const{

    if (!init_success_)
        return;

    if (!isFaceFound(config.limitScope())){
        return;
    }

    // ---------- Prepare measurement ----------

    // Get the best measurement from the entity
    ed::MeasurementConstPtr msr = e->lastMeasurement();
    if (!msr)
        return;

    uint min_x, max_x, min_y, max_y;

    // create a view
    rgbd::View view(*msr->image(), msr->image()->getRGBImage().cols);

    // get color image
    const cv::Mat& color_image = msr->image()->getRGBImage();

    // crop it to match the view
    cv::Mat cropped_image(color_image(cv::Rect(0,0,view.getWidth(), view.getHeight())));

    // initialize bounding box points
    max_x = 0;
    max_y = 0;
    min_x = view.getWidth();
    min_y = view.getHeight();

    cv::Mat mask = cv::Mat::zeros(view.getHeight(), view.getWidth(), CV_8UC1);
    // Iterate over all points in the mask
    for(ed::ImageMask::const_iterator it = msr->imageMask().begin(view.getWidth()); it != msr->imageMask().end(); ++it)
    {
        // mask's (x, y) coordinate in the depth image
        const cv::Point2i& p_2d = *it;

        // paint a mask
        mask.at<unsigned char>(*it) = 255;

        // update the boundary coordinates
        if (min_x > p_2d.x) min_x = p_2d.x;
        if (max_x < p_2d.x) max_x = p_2d.x;
        if (min_y > p_2d.y) min_y = p_2d.y;
        if (max_y < p_2d.y) max_y = p_2d.y;
    }

    cv::Rect bouding_box (min_x, min_y, max_x - min_x, max_y - min_y);

    // ----------------------- Process -----------------------

    std::string name = "";
    double name_confidence = -1;
    std::string gender = "";
    double gender_confidence = -1;
    int age = 0;
    double age_confidence = -1;


    // initialize template
    br::Template entity_tmpl(cropped_image(bouding_box));
    br::Template entity_tmpl2(cropped_image(bouding_box));

    // Enroll templates
    entity_tmpl >> *br_age_estimation;
    entity_tmpl2 >> *br_gender_estimation;
//    entity_tmpl >> *br_face_rec;

    const QPoint firstEye = entity_tmpl.file.get<QPoint>("Affine_0");
    const QPoint secondEye = entity_tmpl.file.get<QPoint>("Affine_1");

    printf("eyes: (%d, %d) (%d, %d)\n", firstEye.x(), firstEye.y(), secondEye.x(), secondEye.y());
    printf("age: %d\n", int(entity_tmpl.file.get<float>("Age")));
    printf("gender: %s\n", qPrintable(entity_tmpl2.file.get<QString>("Gender")));

    // Compare templates
//    float comparisonA = br_face_rec_dist->compare(target, queryA);
    // Scores range from 0 to 1 and represent match probability
//    printf("Genuine match score: %.3f\n", comparisonA);

    age =  int(entity_tmpl.file.get<float>("Age"));
    gender = qPrintable(entity_tmpl2.file.get<QString>("Gender"));



    // ----------------------- Assert results -----------------------

    // create group if it doesnt exist
    if (!config.readGroup("perception_result", tue::OPTIONAL))
    {
        config.writeGroup("perception_result");
    }

    config.writeGroup("open_biometrics_ed");

    config.setValue("label", "");
    config.setValue("score", 0);

    // face recogniton result
    config.writeGroup("openbr_face");
    config.setValue("label", name);
    config.setValue("score", name_confidence);
    config.endGroup();  // close openbr_face group

    // age estimation result
    config.writeGroup("openbr_age");
    config.setValue("label", age);
    config.setValue("score", age_confidence);
    config.endGroup();  // close openbr_age group

    // gender estimation
    config.writeGroup("openbr_gender");
    config.setValue("label", gender);
    config.setValue("score", gender_confidence);
    config.endGroup();  // close openbr_gender group

    config.endGroup();  // close open_biometrics_ed group
    config.endGroup();  // close perception_result group


    if (debug_mode_){
        showDebugWindow(cropped_image(bouding_box),
                        name,
                        name_confidence,
                        gender,
                        gender_confidence,
                        age,
                        age_confidence);
    }
}


// ----------------------------------------------------------------------------------------------------


void OpenBrEd::showDebugWindow(cv::Mat face_img,
                               std::string name,
                               double name_confidence,
                               std::string gender,
                               double gender_confidence,
                               int age,
                               double age_confidence) const{

    int key_press;

    cv::Scalar color_red (0, 0, 255);
    cv::Scalar color_yellow (0, 255, 255);
    cv::Scalar color_green (0, 204, 0);
    cv::Scalar color_gray (100, 100, 100);
    cv::Scalar color_white (255, 255, 255);

    cv::Scalar name_color = color_white;
    cv::Scalar gender_color = color_white;
    cv::Scalar age_color = color_white;

    cv::Mat debug_display(cv::Size(face_img.cols + 60, face_img.rows + 135), CV_8UC1, cv::Scalar(0,0,0));
    cv::Mat debug_roi = debug_display(cv::Rect(30,0, face_img.cols, face_img.rows));
    face_img.copyTo(debug_roi);

    cvtColor(debug_display, debug_display, CV_GRAY2RGB);

/*
    // set colors of the text depending on the score
    if (confidence[EIGEN]/eigen_treshold_ > 0.0 && confidence[EIGEN]/eigen_treshold_ <= 0.8)
        eigen_color = color_green;
    else if (confidence[EIGEN]/eigen_treshold_ > 0.8 && confidence[EIGEN]/eigen_treshold_ <= 1.0)
        eigen_color = color_yellow;
    else if (confidence[EIGEN]/eigen_treshold_ > 1.0)
        eigen_color = color_red;
    else
        eigen_color = color_gray;

    if (confidence[FISHER]/fisher_treshold_ > 0.0 && confidence[FISHER]/fisher_treshold_ <= 0.8)
        fisher_color = color_green;
    else if (confidence[FISHER]/fisher_treshold_ > 0.8 && confidence[FISHER]/fisher_treshold_ <= 1.0)
        fisher_color = color_yellow;
    else if (confidence[FISHER]/fisher_treshold_ > 1.0 )
        fisher_color = color_red;
    else
        fisher_color = color_gray;

    if (confidence[LBPH]/lbph_treshold_ > 0.0 && confidence[LBPH]/lbph_treshold_ <= 0.8)
        lbph_color = color_green;
    else if (confidence[LBPH]/lbph_treshold_ > 0.8 && confidence[LBPH]/lbph_treshold_ <= 1.0)
        lbph_color = color_yellow;
    else if (confidence[LBPH]/lbph_treshold_ > 1.0 )
        lbph_color = color_red;
    else
        lbph_color = color_gray;

    if (confidence[HIST] > 0.0 && confidence[HIST] <= 0.6)
        hist_match_color = color_red;
    else if (confidence[HIST] > 0.6 && confidence[HIST] <= 0.8)
        hist_match_color = color_yellow;
    else if (confidence[HIST] > 0.8)
        hist_match_color = color_green;
    else
        hist_match_color = color_gray;
*/

    std::string info1("Name: " + name + " (" + boost::str(boost::format("%.0f") % name_confidence) + ")");
    std::string info2("Gender: " + gender + " (" + boost::str(boost::format("%.0f") % gender_confidence) + ")");
    std::string info3("Age:  " + boost::str(boost::format("%.0f") % age) + " (" + boost::str(boost::format("%.0f") % age_confidence) + ")");

    // draw text
    cv::putText(debug_display, info1, cv::Point(10 , face_img.rows + 20), 1, 1.1, name_color, 1, CV_AA);
    cv::putText(debug_display, info2, cv::Point(10 , face_img.rows + 40), 1, 1.1, gender_color, 1, CV_AA);
    cv::putText(debug_display, info3, cv::Point(10 , face_img.rows + 60), 1, 1.1, age_color, 1, CV_AA);

    cv::imshow("Open Biometrics ED Output", debug_display);
}


// ----------------------------------------------------------------------------------------------------


bool OpenBrEd::isFaceFound(tue::Configuration config) const{

    double score;
    std::string group_label = "face";

    if (!config.readGroup("perception_result", tue::OPTIONAL))
        return false;

    if (!config.readGroup("face_detector", tue::OPTIONAL))
        return false;

    if (config.value("score", score, tue::OPTIONAL) && config.value("label", group_label, tue::OPTIONAL)){
        return score == 1;
    }
}

// ----------------------------------------------------------------------------------------------------


ED_REGISTER_PERCEPTION_MODULE(OpenBrEd)
