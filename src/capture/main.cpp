#include <k4a/k4a.h>
#include <k4arecord/record.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono;

static void die(const std::string &msg, int code = 1)
{
    std::cerr << msg << std::endl;
    std::exit(code);
}

static bool parse_arg_value(int argc, char **argv, const char *key, std::string &out)
{
    for (int i = 1; i < argc - 1; i++)
    {
        if (std::strcmp(argv[i], key) == 0)
        {
            out = argv[i + 1];
            return true;
        }
    }
    return false;
}

static std::string get_serial(k4a_device_t dev)
{
    char buf[256];
    size_t sz = sizeof(buf);
    if (K4A_FAILED(k4a_device_get_serialnum(dev, buf, &sz)))
    {
        return "unknown_serial";
    }
    return std::string(buf);
}

static std::string make_filename(int index, const std::string &serial)
{
    std::ostringstream oss;
    oss << "k4a_" << index << "_" << serial << ".mkv";
    return oss.str();
}

static void set_manual_exposure_and_gain(k4a_device_t dev,
                                         int32_t exposure_usec,
                                         int32_t gain)
{
    if (K4A_FAILED(k4a_device_set_color_control(dev,
                                               K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                                               K4A_COLOR_CONTROL_MODE_MANUAL,
                                               exposure_usec)))
    {
        die("Failed to set manual exposure. (Is the color camera enabled?)");
    }

    if (K4A_FAILED(k4a_device_set_color_control(dev,
                                               K4A_COLOR_CONTROL_GAIN,
                                               K4A_COLOR_CONTROL_MODE_MANUAL,
                                               gain)))
    {
        die("Failed to set manual gain.");
    }
}

struct DeviceCtx
{
    int index = -1;
    k4a_device_t dev = nullptr;
    std::string serial;

    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

    k4a_record_t rec = nullptr;
    std::string filename;
};

int main(int argc, char **argv)
{
    uint32_t device_count = k4a_device_get_installed_count();
    if (device_count == 0)
    {
        die("No Azure Kinect devices found!");
    }

    // CLI options

    int recording_length_sec = 3;
    int master_index = -1; // auto detects unless you want to override

    // EXPOSURE DEFAULTS!!!
    int32_t exposure_usec = 2500; // reduce motion blur, make frame darker
    int32_t gain = 60; // makes frame lighter but more grainy

    // IR depth delay between master and sub
    int32_t subordinate_delay_usec = 160;

    std::string tmp;
    if (parse_arg_value(argc, argv, "--seconds", tmp))
        recording_length_sec = std::stoi(tmp);

    if (parse_arg_value(argc, argv, "--master-index", tmp))
        master_index = std::stoi(tmp);

    std::string master_serial_arg;
    bool master_serial_provided = parse_arg_value(argc, argv, "--master-serial", master_serial_arg);

    if (parse_arg_value(argc, argv, "--exposure-usec", tmp))
        exposure_usec = std::stoi(tmp);

    if (parse_arg_value(argc, argv, "--gain", tmp))
        gain = std::stoi(tmp);

    if (parse_arg_value(argc, argv, "--sub-delay-usec", tmp))
        subordinate_delay_usec = std::stoi(tmp);

    std::cout << device_count << " device(s) found." << std::endl;

    // Open all devices
    std::vector<DeviceCtx> devices;
    devices.reserve(device_count);

    for (uint32_t i = 0; i < device_count; i++)
    {
        DeviceCtx ctx;
        ctx.index = static_cast<int>(i);

        if (K4A_FAILED(k4a_device_open(i, &ctx.dev)))
        {
            die("Failed to open device " + std::to_string(i));
        }

        ctx.serial = get_serial(ctx.dev);
        ctx.filename = make_filename(ctx.index, ctx.serial);

        std::cout << "Device " << ctx.index << " serial: " << ctx.serial << std::endl;
        devices.push_back(ctx);
    }

    // Determine master camera from sync jack state (master = SYNC OUT connected, SYNC IN disconnected)
    // unless user overrides via --master-index or --master-serial
    auto find_master_by_serial = [&](const std::string &serial) -> int {
        for (auto &d : devices)
            if (d.serial == serial) return d.index;
        return -1;
    };

    auto find_master_by_sync_jack = [&]() -> int {
        int found_index = -1;
        for (auto &d : devices)
        {
            bool sync_in = false, sync_out = false;
            if (K4A_FAILED(k4a_device_get_sync_jack(d.dev, &sync_in, &sync_out)))
            {
                die("Failed to read sync jack state for device " + std::to_string(d.index));
            }

            std::cout << "Device " << d.index
                      << " sync_in=" << (sync_in ? "true" : "false")
                      << " sync_out=" << (sync_out ? "true" : "false")
                      << std::endl;

            if (sync_out && !sync_in)
            {
                if (found_index != -1)
                {
                    die("Multiple master candidates detected (sync_out=true, sync_in=false). "
                        "Fix cabling or pass --master-index/--master-serial.");
                }
                found_index = d.index;
            }
        }
        return found_index;
    };

    if (master_serial_provided)
    {
        master_index = find_master_by_serial(master_serial_arg);
        if (master_index == -1)
        {
            die("Master serial not found among connected devices: " + master_serial_arg);
        }
    }
    else if (master_index == -1)
    {
        master_index = find_master_by_sync_jack();
        if (master_index == -1)
        {
            die("No master detected via sync jacks (need sync_out=true and sync_in=false on exactly one device). "
                "Fix cabling or pass --master-index/--master-serial.");
        }
    }

    if (master_index < 0 || master_index >= static_cast<int>(device_count))
    {
        die("Invalid master index.");
    }

    std::cout << "MASTER device index: " << master_index << std::endl;

    // CONFIGURE DEVICES
    for (auto &d : devices)
    {
        d.config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

        // https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_gabd9688eb20d5cb878fd22d36de882ddb.html
        d.config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;

        // https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_gabc7cab5e5396130f97b8ab392443c7b8.html
        d.config.color_resolution = K4A_COLOR_RESOLUTION_1440P;

        // https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_ga3507ee60c1ffe1909096e2080dd2a05d.html
        d.config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;

        // 5, 15, or 30
        d.config.camera_fps = K4A_FRAMES_PER_SECOND_30;

        d.config.synchronized_images_only = true;

        if (d.index == master_index)
        {
            d.config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
            d.config.subordinate_delay_off_master_usec = 0;
        }
        else
        {
            d.config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
            d.config.subordinate_delay_off_master_usec = subordinate_delay_usec;
        }

        d.config.depth_delay_off_color_usec = 0;
    }

    for (auto &d : devices)
    {
        set_manual_exposure_and_gain(d.dev, exposure_usec, gain);
    }

    for (auto &d : devices)
    {
        if (K4A_FAILED(k4a_record_create(d.filename.c_str(), d.dev, d.config, &d.rec)))
        {
            die("Unable to create recording file: " + d.filename);
        }
        if (K4A_FAILED(k4a_record_write_header(d.rec)))
        {
            die("Unable to write header for: " + d.filename);
        }
    }

    // start the cameras in the order described: subs then master
    for (auto &d : devices)
    {
        if (d.index == master_index)
            continue;

        std::cout << "Starting SUBORDINATE device " << d.index << "..." << std::endl;
        if (K4A_FAILED(k4a_device_start_cameras(d.dev, &d.config)))
        {
            die("Failed to start cameras on subordinate device " + std::to_string(d.index));
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    {
        auto &m = devices[master_index];
        std::cout << "Starting MASTER device " << m.index << "..." << std::endl;
        if (K4A_FAILED(k4a_device_start_cameras(m.dev, &m.config)))
        {
            die("Failed to start cameras on master device " + std::to_string(m.index));
        }
    }

    std::cout << "All devices started. Recording for " << recording_length_sec << "s..." << std::endl;

    const auto end_time = steady_clock::now() + seconds(recording_length_sec);
    const int timeout_ms = 100;

    // capture loop write to .mkv for each camera
    while (steady_clock::now() < end_time)
    {
        for (auto &d : devices)
        {
            k4a_capture_t cap = nullptr;
            k4a_wait_result_t wr = k4a_device_get_capture(d.dev, &cap, timeout_ms);

            if (wr == K4A_WAIT_RESULT_SUCCEEDED)
            {
                if (K4A_FAILED(k4a_record_write_capture(d.rec, cap)))
                {
                    k4a_capture_release(cap);
                    die("Failed to write capture for device " + std::to_string(d.index));
                }
                k4a_capture_release(cap);
            }
            else if (wr == K4A_WAIT_RESULT_TIMEOUT)
            {
                continue;
            }
            else
            {
                die("k4a_device_get_capture() failed on device " + std::to_string(d.index));
            }
        }
    }

    std::cout << "Stopping cameras and closing recordings..." << std::endl;

    // DONE!
    for (auto &d : devices)
    {
        k4a_device_stop_cameras(d.dev);
    }

    for (auto &d : devices)
    {
        (void)k4a_record_flush(d.rec);
        k4a_record_close(d.rec);
        d.rec = nullptr;
    }

    for (auto &d : devices)
    {
        k4a_device_close(d.dev);
        d.dev = nullptr;
    }

    std::cout << "Done. Wrote:" << std::endl;
    for (auto &d : devices)
    {
        std::cout << "  " << d.filename << std::endl;
    }

    return 0;
}
