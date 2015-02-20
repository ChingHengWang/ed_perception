/*
* Author: Luis Fererira
* E-mail: luisfferreira@outlook.com
* Date: January 2015
*/

#ifndef ED_PERCEPTION_OPEN_BIOMETRICS_ED_H_
#define ED_PERCEPTION_OPEN_BIOMETRICS_ED_H_

#include <ed/perception_modules/perception_module.h>

// OpenCV includes
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"

#include <openbr/openbr_plugin.h>

class OpenBrEd : public ed::PerceptionModule
{

    struct FaceFeatures{
        int face_x;
        int face_y;
        int face_height;
        int face_width;
        int first_eye_x;
        int first_eye_y;
        int second_eye_x;
        int second_eye_y;
    };

/*
 * ###########################################
 *  				PRIVATE
 * ###########################################
 */
private:

    // module configuration
    mutable bool init_success_;
    bool debug_mode_;            /*!< Enable debug mode */
    std::string	module_name_;    /*!< Name of the module, for output */
    std::string	module_path_;    /*!< Name of the module, for output */
    std::string debug_folder_;   /*!< Path of the debug folder */

    // open biometrics
    QSharedPointer<br::Transform> br_age_gender_estimat;
    QSharedPointer<br::Transform> br_age_estimation;
    QSharedPointer<br::Transform> br_gender_estimation;
    QSharedPointer<br::Transform> br_face_recogn;
    QSharedPointer<br::Transform> br_face_detect;
    QSharedPointer<br::Distance> br_face_rec_dist;

    void showDebugWindow(cv::Mat face_img,
                         std::string name,
                         double name_confidence,
                         std::string gender,
                         double gender_confidence,
                         int age,
                         double age_confidence) const;

    bool isFaceFound(tue::Configuration config) const;

/*
* ###########################################
*  				    PUBLIC
* ###########################################
*/
public:

    OpenBrEd();

    virtual ~OpenBrEd();

    void loadConfig(const std::string& config_path);

    void process(ed::EntityConstPtr e, tue::Configuration& result) const;

    void configure(tue::Configuration config);

};

#endif
