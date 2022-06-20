/*
* To be placed in openpose/examples/user_code
* See OpenPose's documentation for more info at https://github.com/CMU-Perceptual-Computing-Lab/openpose
*
* It reads the frames from the input video, process them and display them with the pose (and optionally hand and face) keypoints.
* It creates an output csv file in which every line is formed by:
*   - frame number
*   - keypoint name
*   - X and Y coordinates
*   - model's confidence
*
* At the end of computation it prints on screen the avarage FPS, the avarage overall confidence
* and the avarage confidence considering only the index and the nose.
*
* it includes all the OpenPose configuration flags (enable/disable hand, face, no_display, etc.).
*/

// Third-party dependencies
#include <opencv2/opencv.hpp>
#include <chrono>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_dir,                "examples/media/",
    "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// paths to the input video and to the output csv file
const char video_path[] = "/home/parco04/OneDrivePARCO/General/Video/Index-Nose Pose Estimation/video0001-0090.avi";
const char output_csv_path[] = "/home/parco04/op_output.csv";

// This worker will just read and return all the jpg files in a directory
bool display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKPs: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        return (key == 27);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return true;
    }
}


double printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, unsigned frameNumber, std::ofstream *outcsv, double *avgConfIndexNose)
{
    double avgFrameConfidence = 0.0f;

    try
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            /*
            // I wanted to do this:
            // retrieve the used model
            const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));

            // use the model to retrieve the body parts map
            const std::map<unsigned int, std::string>& bodyPartsMap = op::getPoseBodyPartMapping(poseModel);

            // but the model will always be BODY_25 because the other models are currently marked
            // as 'Experimental' and they will not be retrieved by any function, so I did in the following way...
            */
            const std::map<unsigned int, std::string>& bodyPartsMap = op::getPoseBodyPartMapping(op::PoseModel(14));
            // 14 is the index corresponding to POSE_BODY_135_BODY_PARTS: a map with all the 135 body parts
            // Instad of passing the model to getPoseBodyPartMapping, I force the function to returne the complete map

            std::string valueToPrint;
            double totSize = 0.0f;
            const double confThresh = 0.30f;     // confidence threshold
            const auto& poseKPs = datumsPtr->at(0)->poseKeypoints;
            const auto& faceKPs = datumsPtr->at(0)->faceKeypoints;
            const auto& handsKPs = datumsPtr->at(0)->handKeypoints;
            const int& imWidth = datumsPtr->at(0)->cvInputData.cols();
            const int& imHeight = datumsPtr->at(0)->cvInputData.rows();

            /* assuming there's only one person (as we need) */

            for (int bodyPart = 0; bodyPart < poseKPs.getSize(1); bodyPart++) {
                bool foundOutOfBoundaries = false;
                valueToPrint += std::to_string(frameNumber) + ";";
                valueToPrint += bodyPartsMap.at(bodyPart) + ";";

                const auto size = poseKPs.getSize(2);
                for (int xyscore = 0; xyscore < size; xyscore++) {
                    /* Check if the current point's coordinates are withing the image boundaries.
                    This happens for a couple of KPs and it creates problem with INDE_performance_test */
                    if (poseKPs[ {0, bodyPart, 0} ] > imWidth || poseKPs[ {0, bodyPart, 1} ] > imHeight ||
                        poseKPs[ {0, bodyPart, 0} ] < 0 || poseKPs[ {0, bodyPart, 1} ] < 0)
                    {
                        foundOutOfBoundaries = true;
                        break;
                    }

                    valueToPrint += std::to_string(poseKPs[ {0, bodyPart, xyscore} ]) + ";";

                    // if foundOutOfBoundaries, we won't reach this if statement
                    if (xyscore == size - 1 && poseKPs[{0, bodyPart, xyscore}] >= confThresh) {
                        avgFrameConfidence += poseKPs[{0, bodyPart, xyscore}];
                        totSize++;
                    }
                }
                if (! foundOutOfBoundaries) {
                    valueToPrint = valueToPrint.substr(0, valueToPrint.size() - 1); // delete last semicolon (needed for INDE_performance_test)
                    //op::opLog(valueToPrint, op::Priority::High);
                    *outcsv << valueToPrint << '\n';
                }
                valueToPrint = "";
            }

            if (FLAGS_face) {
                for (int facePart = 0; facePart < faceKPs.getSize(1); facePart++) {
                    valueToPrint += std::to_string(frameNumber) + ";";
                    valueToPrint += bodyPartsMap.at(facePart + op::F135) + ";";

                    const auto size = faceKPs.getSize(2);
                    for (int xyscore = 0; xyscore < faceKPs.getSize(2); xyscore++) {
                        valueToPrint += std::to_string(faceKPs[ {0, facePart, xyscore} ]) + ";";

                        if (xyscore == size - 1 && faceKPs[{0, facePart, xyscore}] >= confThresh) {
                            avgFrameConfidence += faceKPs[{0, facePart, xyscore}];
                            totSize++;
                            if (facePart == 30) // 30 is the number of NoseUpper3, the nose tip
                                *avgConfIndexNose += faceKPs[{0, facePart, xyscore}];
                        }
                    }
                    valueToPrint = valueToPrint.substr(0, valueToPrint.size() - 1);
                    //op::opLog(valueToPrint, op::Priority::High);
                    *outcsv << valueToPrint << '\n';
                    valueToPrint = "";
                }
            }

            if (FLAGS_hand) {
                for (short handNumber = 0; handNumber < 2; handNumber++) {
                    for (int handPart = 0; handPart < handsKPs[handNumber].getSize(1) /*- 1*/; handPart++) {
                        valueToPrint += std::to_string(frameNumber) + ";";
                        valueToPrint += bodyPartsMap.at(handPart + op::H135 + 20 * handNumber) + ";";

                        const auto size = handsKPs[handNumber].getSize(2);
                        for (int xyscore = 0; xyscore < handsKPs[handNumber].getSize(2); xyscore++) {
                            valueToPrint += std::to_string(handsKPs[handNumber][{0, handPart, xyscore}]) + ";";

                            if (xyscore == size - 1 && handsKPs[handNumber][{0, handPart, xyscore}] >= confThresh) {
                                avgFrameConfidence += handsKPs[handNumber][{0, handPart, xyscore}];
                                totSize++;
                                // if we're on the left index finger tip...
                                if (handNumber == 0 && handPart == 7)
                                    *avgConfIndexNose += handsKPs[handNumber][{0, handPart, xyscore}];
                            }
                        }
                        valueToPrint = valueToPrint.substr(0, valueToPrint.size() - 1);
                        //op::opLog(valueToPrint, op::Priority::High);
                        *outcsv << valueToPrint << '\n';
                        valueToPrint = "";
                    }
                }
            }

            return avgFrameConfidence / totSize;
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }

    return avgFrameConfidence;  // in case of exception or datumsPtr == nullptr, returns 0
}


void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
            (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}


int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // create VideoCapture object to read the video
        cv::VideoCapture videocap(video_path);
        if (!videocap.isOpened())
            op::opLog("Error opening the video", op::Priority::High);

        // used for fps evaluation
        typedef std::chrono::high_resolution_clock HighResCK;
        typedef std::chrono::duration<double> duration;

        cv::Mat frame;
        unsigned frameNumber = 0;
        double avgFPS = 0.0f;
        double avgConf = 0.0f;
        double avgConfIndexNose = 0.0f; // Avarage frame confidence only for index and nose

        // open csv file to save keypoints data
        std::ofstream outcsv;
        outcsv.open(output_csv_path, std::ios::out | std::ios::trunc);

        while (1) {
            if (!videocap.read(frame))  // read a frame of the video and put it into "frame"
                break;

            // process the image and compute the time taken
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(frame);
            auto startTime = HighResCK::now();

            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);

            duration elabTime = HighResCK::now() - startTime;
            avgFPS += elabTime.count();

            if (datumProcessed != nullptr) {
                avgConf += printKeypoints(datumProcessed, ++frameNumber, &outcsv, &avgConfIndexNose);
                if (!FLAGS_no_display)
                    display(datumProcessed);
            }
        }

        outcsv.close();

        avgFPS /= (double) frameNumber;
        avgFPS = /*(double)*/ 1.0f / avgFPS;
        avgConf /= (double) frameNumber;
        avgConfIndexNose /= (double)(2 * frameNumber);  // 2 is the number of body parts included in the measurement

        // Measuring total time, avarage FPS and avarage confidence
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);
        op::opLog("Avarage FPS: " + std::to_string(avgFPS), op::Priority::High);
        op::opLog("Avarage confidence: " + std::to_string(avgConf), op::Priority::High);
        op::opLog("Avarage confidence for index and nose: " + std::to_string(avgConfIndexNose), op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}


int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
