import subprocess


print("\n========================================")
print("      CITY COMPATIBILITY ENGINE")
print("========================================")


print("\n[1/3] Validating data...")
subprocess.run([
    r"D:\Anaconda3\python.exe",
    "src/validate_data.py"
], check=True)


print("\n[2/3] Normalizing data...")
subprocess.run([
    r"D:\Anaconda3\python.exe",
    "src/normalize_features.py"
], check=True)


print("\n[3/3] Running recommendation engine...")
subprocess.run([
    r"D:\Anaconda3\python.exe",
    "src/hybrid.py"
], check=True)


print("\n========================================")
print("          ENGINE COMPLETE ")
print("========================================")