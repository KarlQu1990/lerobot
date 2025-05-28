import platform
import subprocess
from pathlib import Path
from threading import RLock

import pandas as pd

if platform.system() == "Windows":
    from comtypes import GUID
    from comtypes.persist import IPropertyBag
    from pygrabber.dshow_graph import FilterGraph
    from pygrabber.dshow_ids import DeviceCategories

    COM_OUTPUT_PATH = Path(__file__).parents[3] / "data" / "com_devices.html"
    USBDEVIEW_PATH = Path(__file__).parents[3] / "libs" / "usbdeview" / "USBDeview.exe"

    class USBDeviceManager(object):
        _lock = RLock()
        _instance = None

        _devices = {}
        _loaded = False

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                with cls._lock:
                    if cls._instance is None:
                        cls._instance = super().__new__(cls, *args, **kwargs)

            return cls._instance

        @property
        def devices(self):
            return self._devices

        def load(self) -> "USBDeviceManager":
            if self._loaded:
                return self

            with self._lock:
                if self._loaded:
                    return self

                COM_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

                subprocess.run(f"{USBDEVIEW_PATH} /DisplayDisconnected 0 /shtml {COM_OUTPUT_PATH}", check=True)
                print(f"export COM device info to: {COM_OUTPUT_PATH}")

                tables = pd.read_html(COM_OUTPUT_PATH)
                table = tables[0]

                table = table.loc[~pd.isna(table["Friendly Name"]), :]
                devices = table.set_index("Friendly Name")[["Instance ID", "Drive Letter"]].to_dict(orient="index")

                self._devices = devices
                self._loaded = True

            return self

    def get_camera_device_paths() -> list[str]:
        def get_moniker_instance_id(moniker) -> str:
            property_bag = moniker.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)
            return property_bag.Read("DevicePath", pErrorLog=None)

        def get_input_device_paths(graph: FilterGraph):
            filter_enumerator = graph.system_device_enum.system_device_enum.CreateClassEnumerator(
                GUID(DeviceCategories.VideoInputDevice), dwFlags=0
            )
            result: list[str] = []
            try:
                moniker, count = filter_enumerator.Next(1)
            except ValueError:
                return result
            while count > 0:
                result.append(get_moniker_instance_id(moniker))
                moniker, count = filter_enumerator.Next(1)
            return result

        graph = FilterGraph()
        device_paths = get_input_device_paths(graph)

        return device_paths

else:

    class USBDeviceManager(object):
        _lock = RLock()
        _instance = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                with cls._lock:
                    if cls._instance is None:
                        cls._instance = super().__new__(cls, *args, **kwargs)

            return cls._instance

        def load(self) -> "USBDeviceManager":
            return self
