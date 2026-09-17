import ids_peak.ids_peak as ids_peak

ids_peak.Library.Initialize()
dm = ids_peak.DeviceManager.Instance()
dm.Update()
dev     = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
nodemap = dev.RemoteDevice().NodeMaps()[0]

# Check current state and availability
node = nodemap.FindNode("SensorShutterMode")
print("Current mode:", node.CurrentEntry().SymbolicValue())
for e in node.Entries():
    try:
        name = e.SymbolicValue()
        avail = e.AccessStatus()
        print(f"  {name}: {avail}")
    except Exception:
        pass

# Try setting GlobalReset
try:
    nodemap.FindNode("SensorShutterMode").SetCurrentEntry("GlobalReset")
    print("GlobalReset SET OK")
except Exception as ex:
    print(f"GlobalReset failed: {ex}")

    # Try again after switching to triggered mode — GlobalReset availability
    # is invalidated by TriggerMode in the XML
    print("Trying in triggered mode...")
    nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
    nodemap.FindNode("TriggerMode").SetCurrentEntry("On")
    try:
        nodemap.FindNode("SensorShutterMode").SetCurrentEntry("GlobalReset")
        print("GlobalReset SET OK in triggered mode")
    except Exception as ex2:
        print(f"GlobalReset failed in triggered mode too: {ex2}")

ids_peak.Library.Close()