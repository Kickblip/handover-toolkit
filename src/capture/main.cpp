#include <iostream>
#include <k4a/k4a.h>
#include "recorder.h"

int main()
{
    // get the number of connected devices
    uint32_t device_count = k4a_device_get_installed_count();
    if (device_count == 0)
    {
        std::cerr << "No Azure Kinect devices found!" << std::endl;
        return 1;
    }

    std::cout << device_count << " device(s) found. Using the first one." << std::endl;

    // open the first device
    k4a_device_t device;
    if (K4A_FAILED(k4a_device_open(0, &device)))
    {
        std::cerr << "Failed to open device 0!" << std::endl;
        return 1;
    }

    k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    device_config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    device_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    device_config.depth_mode = K4A_DEPTH_MODE_OFF;  // No depth
    device_config.camera_fps = K4A_FRAMES_PER_SECOND_30;

    // record for short burst
    const int recording_length = 3;
    char recording_filename[] = "azure_kinect_recording.mkv";
    bool record_imu = false;  // imu data not necessary
    int32_t exposure_value = defaultExposureAuto;
    int32_t gain_value = defaultGainAuto;

    // call the recording function
    int result = do_recording(0, recording_filename, recording_length, &device_config, record_imu, exposure_value, gain_value);

    k4a_device_close(device);

    return result;
}
