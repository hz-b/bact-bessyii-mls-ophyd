from bact_mls_ophyd.devices.pp.bpm import BPM

bpm = BPM("BPMZ1X003GP", name="bpm")
if not bpm.connected:
    bpm.wait_for_connection()

for key, val in bpm.describe().items():
    print(key)
    print(val, "\n")
#print(bpm.describe())
stat = bpm.trigger()
stat.wait(3)
data = bpm.read()
print("# ---- data")
print(data)
print("# ---- end data ")
