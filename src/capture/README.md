
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

Try this where 128 = 32*4 (4 cameras):

```
cat /sys/module/usbcore/parameters/usbfs_memory_mb
sh -c "echo 128 > /sys/module/usbcore/parameters/usbfs_memory_mb"
```

As listed in [Issue 485](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/485)