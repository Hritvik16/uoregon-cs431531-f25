import os, re, sys, matplotlib.pyplot as plt

folder = "output"
if not os.path.isdir(folder):
    print("ERROR: folder", folder, "does not exist")
    sys.exit(1)

files_all = sorted(os.listdir(folder))
print("FOUND FILES:", files_all)

files = []
for fn in files_all:
    m = re.search(r"(\d+)_CPU", fn)
    if m:
        files.append(fn)
    else:
        print("SKIP (no cpu match):", fn)
files = sorted(files, key=lambda x: int(re.search(r"(\d+)_CPU", x).group(1)))
print("PARSABLE FILES (sorted):", files)

cpus = []
serial_times = []
nlogn_times = []
n_times = []

for filename in files:
    path = os.path.join(folder, filename)
    print("\n--- Processing file:", filename, "path:", path)
    try:
        with open(path) as f:
            raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
    except Exception as e:
        print("FAILED to open", path, ":", e)
        continue
    print("RAW LINES:")
    for i,ln in enumerate(raw_lines):
        print(i, ":", ln)
    if len(raw_lines) < 3:
        print("WARN: less than 3 lines in file, skipping")
        continue
    m = re.search(r"(\d+)_CPU", filename)
    if not m:
        print("WARN: couldn't extract cpu count from filename, skipping")
        continue
    cpu = int(m.group(1))
    def extract_time(line):
        nums = re.findall(r"([\d]*\.\d+|\d+)(?=\s*\(s\))", line)
        if nums:
            val = float(nums[-1])
            print("EXTRACT (with (s)) from:", line, "=>", val)
            return val
        nums2 = re.findall(r"([\d]*\.\d+|\d+)", line)
        if nums2:
            val = float(nums2[-1])
            print("EXTRACT (fallback) from:", line, "=>", val)
            return val
        print("ERROR: no numeric token found in line:", line)
        raise ValueError("no numeric token")
    try:
        t_serial = extract_time(raw_lines[0])
        t_nlogn = extract_time(raw_lines[1])
        t_n = extract_time(raw_lines[2])
    except Exception as e:
        print("ERROR parsing times in", filename, ":", e)
        continue
    print("PARSED:", filename, "cpu=", cpu, "serial=", t_serial, "nlogn=", t_nlogn, "n=", t_n)
    cpus.append(cpu)
    serial_times.append(t_serial)
    nlogn_times.append(t_nlogn)
    n_times.append(t_n)

print("\nSUMMARY PARSED")
print("cpus:", cpus)
print("serial_times:", serial_times)
print("nlogn_times:", nlogn_times)
print("n_times:", n_times)

if not (len(cpus) and len(serial_times) == len(cpus) == len(nlogn_times) == len(n_times)):
    print("ERROR: parsed arrays length mismatch or empty")
    sys.exit(2)

order = sorted(range(len(cpus)), key=lambda i: cpus[i])
cpus = [cpus[i] for i in order]
serial_times = [serial_times[i] for i in order]
nlogn_times = [nlogn_times[i] for i in order]
n_times = [n_times[i] for i in order]

print("\nFINAL ORDERED")
print("cpus:", cpus)
print("serial_times:", serial_times)
print("nlogn_times:", nlogn_times)
print("n_times:", n_times)

plt.figure(figsize=(8,5))
plt.plot(cpus, serial_times, 'o-', label='Serial O(N)')
plt.plot(cpus, nlogn_times, 'o-', label='Parallel O(NlogN)')
plt.plot(cpus, n_times, 'o-', label='Parallel O(N)')
plt.xlabel("Number of CPUs")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs Number of CPUs")
# plt.xscale("log", base=2)
plt.xscale("linear")
plt.legend()
plt.grid(True)
out1 = "execution_time_vs_cpus.png"
plt.savefig(out1, dpi=300)
print("WROTE:", out1)

baseline_serial = None
if 1 in cpus:
    baseline_serial = serial_times[cpus.index(1)]
    print("BASELINE serial @1CPU =", baseline_serial)
else:
    baseline_serial = serial_times[0]
    print("NOTE: 1 CPU not present, using first serial value as baseline =", baseline_serial)

speedup_nlogn = [baseline_serial / t for t in nlogn_times]
speedup_n = [baseline_serial / t for t in n_times]

plt.figure(figsize=(8,5))
plt.plot(cpus, speedup_nlogn, 'o-', label='Speedup O(NlogN)')
plt.plot(cpus, speedup_n, 'o-', label='Speedup O(N)')
plt.xlabel("Number of CPUs")
plt.ylabel("Speedup (serial@1 / parallel)")
plt.title("Speedup vs Number of CPUs")
# plt.xscale("log", base=2)
plt.xscale("linear")

plt.legend()
plt.grid(True)
out2 = "speedup_vs_cpus.png"
plt.savefig(out2, dpi=300)
print("WROTE:", out2)

plt.show()
