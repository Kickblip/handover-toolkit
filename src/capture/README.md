
## Build commands
```
cmake -S src/capture -B build/capture -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build/capture -j
cmake --install build/capture
```

*Make sure .local/bin is on your path*

Alternatively, run the binary directly:
`./build/capture/htkrecorder`


Sometimes the Azure Kinects use too much memory when multiple are plugged in.

Try this:

```
cat /sys/module/usbcore/parameters/usbfs_memory_mb
sh -c "echo 32 > /sys/module/usbcore/parameters/usbfs_memory_mb"
```

As listed in [Issue 485](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/485)